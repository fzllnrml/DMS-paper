"""Confirmation orchestration only; tiny random models and synthetic fixtures, no training."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
import yaml

from test_mnist_controls import fixture, fake_mnist
from src.mnist_experiments.common import save_pt, fingerprint, write_json, code_digest
from src.mnist_experiments.publication import definition_hash, definitions
from src.mnist_controls import execution
from src.mnist_controls.protocol import load_config as pilot_config, job_specs, QUALITY, METRICS, BASELINES
from src.mnist_controls.confirmation import (load_config, dry_run, model_configs, execute,
    METHOD_PARAMETERS, INFERENCE)
from src.mnist_controls.confirmation_analysis import aggregate_confirmation, intervals
from src.mnist_controls.artifacts import read_json, run_job


def tree_hash(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def tiny_config(root):
    tiny=fixture(root)
    c=load_config()
    for key in ('population_size','pool_size','batch_size','evaluation','device','publication_root'):
        c[key]=deepcopy(tiny[key])
    c.update(output_root=str(root/'confirmation'),pilot_root=str(root/'pilot'))
    c['checkpoints']={}
    state=torch.load(tiny['checkpoint'],weights_only=True)
    for seed in c['model_seeds']:
        state['config']['seed']=seed
        path=root/f'model{seed}.pt';save_pt(path,state);c['checkpoints'][seed]=str(path)
    return c


def artifact_fixture(c):
    """Synthetic result fixtures, never treated as experimental evidence."""
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=True)
    context=dict(config=c,implementation='unit-test-fixture')
    sig=fingerprint(context)
    write_json(root/'manifest.json',dict(context=context,signature=sig,jobs=list(job_specs(c))))
    for seed in c['search_seeds']:
        common=dict(model_seed=c['model_seed'],search_seed=seed,checkpoint_sha256=f"model{c['model_seed']}",
            evaluator_sha256='same-evaluator',evaluation_cache_key='same-cache',
            initial_gaussian_prefix_sha256=f'noise{seed}',initial_ddim_prefix_sha256=f'init{seed}',
            context_signature=sig)
        def row(method):
            return dict(common,method=method,pixel_knn_mean=.2 if method=='rms_v3f2' else .3,
                feature_knn_mean=4. if method=='rms_v3f2' else 6.,pixel_support_percentile=50.,
                feature_support_percentile=50.,class_coverage=10,class_entropy=2.2,nfe=13440,
                newly_executed_nfe=0)
        folder=root/f"model{c['model_seed']}_seed{seed}"
        run_job(folder/'inputs',sig,c,lambda:(dict(method='inputs',newly_executed_nfe=0,
            baseline_rows=[row(m) for m in BASELINES]),{}, {},common))
        for method in c['methods']+c['diagnostics']:
            summary=row(method);metrics={}
            if method not in QUALITY:
                summary.update(diagnostic_nfe=6400,stability_summary=[dict(scale=s,valid_fraction=1.,
                    median_rho=.7,mean_rho=.7,fraction_rho_lt_1=1.,fraction_rho_le_09=1.,
                    median_reference_drift=.005) for s in (.02,.05)])
                metrics={'per_candidate':[]}
            run_job(folder/method,sig,c,lambda s=summary,m=metrics:(s,m,{},common))
    return root


class ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_config_counts_and_frozen_protocol(self):
        c=load_config();p=pilot_config()
        self.assertEqual(c['model_seeds'],[43,44]);self.assertEqual(c['search_seeds'],list(range(5)))
        for key in ('population_size','pool_size','batch_size','methods','diagnostics','evaluation'):
            self.assertEqual(c[key],p[key])
        self.assertEqual(c['method_parameters'],METHOD_PARAMETERS);self.assertEqual(c['inference'],INFERENCE)
        # Existing pilot scientific code and YAML are byte-for-byte preserved.
        for path,expected in {
            'configs/mnist_controls.yaml':'1e8a5bed7332ac7178c19d898fbd181697ef16f4fc76e71fca18a064896e678c',
            'src/mnist_controls/controls.py':'18aac46ca5da04e1c4bd6cc10704a0f4d8d9c2f3ab4874c6cdd6672a7f8e0ffe'}.items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),expected)
        with TemporaryDirectory() as tmp:
            c['publication_root']=tmp
            with patch('torch.load',side_effect=AssertionError('dry run must not load weights')):
                plan=dry_run(c)
            self.assertEqual((plan['quality_jobs'],plan['stability_jobs'],plan['dependency_jobs']),(30,20,10))
            self.assertEqual(len(plan['jobs']),50);self.assertEqual(len({j['id'] for j in plan['jobs']}),50)
            self.assertEqual(plan['new_job_nfe'],332800)
            self.assertEqual(plan['total_new_nfe_upper'],537200)
            self.assertEqual(plan['fair_controls_and_diagnostic_nfe'],530800)
            self.assertEqual(plan['reused_nfe'],198000)
            for child in model_configs(c):
                self.assertEqual(child['model_seed'],43 if '43' in child['checkpoint'] else 44)
            path=Path(tmp)/'bad.yaml'
            c['method_parameters']['low_noise_only']['alpha'][0]=.04
            path.write_text(yaml.safe_dump(c))
            with self.assertRaisesRegex(ValueError,'Frozen'):load_config(path)

    def test_source_discovery_is_model_aware_and_metadata_only(self):
        c=load_config()
        with TemporaryDirectory() as tmp:
            c['publication_root']=tmp
            for model in (42,43,44):
                for seed in range(5):
                    root=Path(tmp)/f'{model}_{seed}';root.mkdir()
                    (root/'resolved_config.yaml').write_text(yaml.safe_dump(dict(model_seed=model,search_seed=seed,population_size=128)))
                    (root/'metadata.json').write_text('{}')
                    for method in ('matched_ddim','rms_v3f2'):
                        target=root/method;target.mkdir()
                        for name in ('arrays.pt','summary.json','complete.json'):(target/name).write_text('presence only')
            with patch('torch.load',side_effect=AssertionError('no tensors in dry-run')):
                p=dry_run(c)
            self.assertEqual(p['total_new_nfe_upper'],332800)
            self.assertEqual(p['dependency_nfe_upper'],0)
            self.assertEqual(len(p['publication_sources']),10)
            self.assertTrue(all(str(r['model_seed']) in Path(r['path']).name for r in p['publication_sources']))
            (Path(tmp)/'44_4/matched_ddim/arrays.pt').unlink()
            self.assertEqual(dry_run(c)['total_new_nfe_upper'],353240)

    @patch('src.mnist_experiments.data.MNIST',side_effect=fake_mnist)
    def test_tiny_two_models_all_seeds_resume_and_hash_safety(self,_):
        with TemporaryDirectory() as tmp:
            c=tiny_config(Path(tmp))
            original=execution.select_representatives;calls=[]
            def fail_once(pool,count):
                calls.append(1)
                if len(calls)==1:raise RuntimeError('injected confirmation selector failure')
                return original(pool,count)
            with patch('src.mnist_controls.execution.select_representatives',side_effect=fail_once):
                result=execute(c)
            self.assertFalse(result['complete'])
            root=Path(c['output_root']);status=read_json(root/'summary/job_status.json')
            self.assertEqual(len(status),60)
            failed=[r for r in status if r['status']=='failed']
            self.assertEqual(len(failed),1);self.assertEqual(failed[0]['model_seed'],43)
            raw=read_json(root/'summary/raw_runs.json')
            self.assertEqual(len(raw),60)
            self.assertEqual({(r['model_seed'],r['search_seed']) for r in raw},{(m,s) for m in (43,44) for s in range(5)})
            done=root/'model44/model44_seed4/ddim_105/complete.json'
            before=(done.read_bytes(),done.stat().st_mtime_ns)
            self.assertTrue(execute(c,resume=True)['complete'])
            self.assertEqual(before,(done.read_bytes(),done.stat().st_mtime_ns))
            with patch('src.mnist_controls.execution.load_backend',side_effect=AssertionError('completed jobs must skip')):
                self.assertTrue(execute(c,resume=True)['complete'])
            with self.assertRaises(FileExistsError):execute(c)
            bad=deepcopy(c);bad['search_seeds']=[0]
            with self.assertRaisesRegex(ValueError,'context changed'):execute(bad,resume=True)
            # Wrong model checkpoint is rejected by the shared backend, not silently relabeled.
            child=list(model_configs(c))[0]
            badchild=deepcopy(child);badchild['checkpoint']=c['checkpoints'][44]
            with self.assertRaisesRegex(ValueError,'model seed43'):
                execution.load_backend(badchild,execution.dependency_context(badchild))
            # The same publication importer accepts seed43 and rejects another model.
            source=root/'model43/model43_seed0/inputs'
            data=torch.load(source/'arrays.pt',weights_only=True)
            prov=read_json(source/'metadata.json')
            old=Path(tmp)/'legacy43';old.mkdir()
            oldcfg=dict(model_seed=43,search_seed=0,population_size=2,initial_noise_seed=0,
                        batch_size=2,ddim_steps=50,evaluation=child['evaluation'])
            (old/'resolved_config.yaml').write_text(yaml.safe_dump(oldcfg))
            meta=dict(signature='test-legacy43',definition_hash=definition_hash(),
                implementation_sha256=code_digest(),checkpoint_sha256=prov['checkpoint_sha256'],
                evaluator_sha256=prov['evaluator_sha256'])
            write_json(old/'metadata.json',meta)
            for method in ('matched_ddim','rms_v3f2'):
                dest=old/method;dest.mkdir()
                row=next(r for r in read_json(source/'summary.json')['baseline_rows'] if r['method']==method)
                row.update(model_seed=43,population_size=2,initial_noise_seed=0,
                           definition_hash=definition_hash(),resolved_method_definition=definitions()[method])
                write_json(dest/'summary.json',row)
                write_json(dest/'complete.json',dict(signature=meta['signature'],summary=row))
                initial=data['pool'] if method=='matched_ddim' else data['initial']
                save_pt(dest/'arrays.pt',dict(initial_noise=data['noise'][:len(initial)],initial=initial,
                    candidates=data['pool'] if method=='matched_ddim' else data['full']))
            context=execution.dependency_context(child)
            pool,initial,full,_=execution.load_publication_inputs(old,child,context,0,data['noise'])
            self.assertTrue(torch.equal(pool,data['pool']));self.assertTrue(torch.equal(full,data['full']))
            oldcfg['model_seed']=44;(old/'resolved_config.yaml').write_text(yaml.safe_dump(oldcfg))
            with self.assertRaisesRegex(ValueError,'model_seed'):
                execution.load_publication_inputs(old,child,context,0,data['noise'])
            state=torch.load(c['checkpoints'][44],weights_only=True);state['changed']=True;save_pt(c['checkpoints'][44],state)
            result=execute(c,resume=True)
            self.assertFalse(result['complete'])
            self.assertEqual(len([r for r in result['failed_or_missing'] if r['model_seed']==44]),30)

    def test_read_only_combined_aggregation_and_failures(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);c=load_config();c.update(output_root=str(root/'confirm'),pilot_root=str(root/'pilot'))
            pc=pilot_config();pc['output_root']=c['pilot_root']
            pilot=artifact_fixture(pc)
            (pilot/'summary').mkdir();(pilot/'summary/raw_runs.csv').write_text('original pilot CSV\n')
            before=tree_hash(pilot)
            for child in model_configs(c):artifact_fixture(child)
            manifest=dict(config=c,implementation='fixture')
            write_json(Path(c['output_root'])/'confirmation_manifest.json',dict(signature=fingerprint(manifest),**manifest))
            report=aggregate_confirmation(c)
            self.assertTrue(report['complete']);self.assertTrue(report['combined_complete'])
            self.assertEqual(tree_hash(pilot),before)
            out=Path(c['output_root'])/'summary'
            self.assertEqual(len(read_json(out/'raw_runs.json')),60)
            combined=read_json(out/'combined/raw_runs.json')
            self.assertEqual(len(combined),78)
            self.assertEqual({r['model_seed'] for r in combined},{42,43,44})
            self.assertEqual({r['phase'] for r in combined},{'pilot','confirmation'})
            ci=[r for r in read_json(out/'paired_comparisons.json') if r['record_type']=='confirmation_interval']
            self.assertEqual(len(ci),6);self.assertTrue(all(r['n_complete']==10 for r in ci))
            decision=read_json(out/'decisions.json')
            self.assertTrue(all(r['search_confirmed'] for r in decision['controls'].values()))
            self.assertTrue(decision['stability_confirmed'])
            # Corrupt result is retained explicitly, never silently removed from means/CI.
            (Path(c['output_root'])/'model44/model44_seed4/ddim_105/metrics.json').write_text('{"changed":true}')
            report=aggregate_confirmation(c);self.assertFalse(report['complete'])
            raw=read_json(out/'raw_runs.json');self.assertEqual(len(raw),60)
            self.assertEqual(len([r for r in raw if r['status']=='corrupt']),1)
            ci=[r for r in read_json(out/'paired_comparisons.json') if r['record_type']=='confirmation_interval' and r['method']=='ddim_105']
            self.assertTrue(all(r['status']=='incomplete' and r['n_complete']==9 and 'ci_lower' not in r for r in ci))
            self.assertEqual(tree_hash(pilot),before)

    def test_missing_models_and_bootstrap_unit(self):
        with TemporaryDirectory() as tmp:
            c=load_config();c.update(output_root=str(Path(tmp)/'confirmation'),pilot_root=str(Path(tmp)/'pilot'))
            root=Path(c['output_root']);root.mkdir()
            identity=dict(config=c,implementation='fixture')
            write_json(root/'confirmation_manifest.json',dict(signature=fingerprint(identity),**identity))
            result=aggregate_confirmation(c)
            self.assertFalse(result['complete']);self.assertEqual(len(result['failed_or_missing']),60)
            self.assertEqual(len(read_json(root/'summary/raw_runs.json')),60)
        pairs=[dict(record_type='pair',method=m,metric=k,model_seed=model,search_seed=s,
                    status='complete',difference=d)
               for m in QUALITY for k in ('pixel_knn_mean','feature_knn_mean')
               for model,d in ((43,.3),(44,.7)) for s in range(5)]
        a=intervals(pairs,[43,44],list(range(5)))
        self.assertEqual(a,intervals(pairs,[43,44],list(range(5))))
        self.assertTrue(all(r['difference']==.5 and r['ci_lower']==.5 and r['ci_upper']==.5 for r in a))
        self.assertTrue(all(r['interval_level']==1-.05/6 for r in a))

    def test_preparation_failure_and_output_isolation(self):
        with TemporaryDirectory() as tmp:
            c=load_config();c.update(output_root=str(Path(tmp)/'confirmation'),pilot_root=str(Path(tmp)/'pilot'))
            with patch('src.mnist_controls.execution.dependency_context',side_effect=FileNotFoundError('missing checkpoint')):
                result=execute(c)
            self.assertFalse(result['complete'])
            self.assertEqual(len(result['failed_or_missing']),60)
            self.assertTrue(all(r['status']=='failed' and 'missing checkpoint' in r['error'] for r in result['failed_or_missing']))
            self.assertFalse(Path(c['pilot_root']).exists())
            bad=deepcopy(c);bad['output_root']=str(Path(c['pilot_root'])/'nested')
            with self.assertRaisesRegex(ValueError,'disjoint'):execute(bad)


if __name__=='__main__':unittest.main()
