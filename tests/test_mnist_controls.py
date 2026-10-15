"""No trained weights/downloads: analytic maps, fake MNIST and a tiny real U-Net."""
from copy import deepcopy
import csv
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import yaml
from src.mnist import MNISTDiffusion
from src.mnist_experiments.common import CountedModel, gaussian_noise, save_pt, code_digest
from src.mnist_experiments.publication import definitions, definition_hash
from src.mnist_experiments.evaluator import EvaluatorCNN
from src.mnist_controls.controls import (ddim_indices, ddim_105, low_noise_only,
    select_representatives, contraction_diagnostic, stability_v2)
from src.mnist_controls.protocol import load_config, plan, tensor_hash, QUALITY
from src.mnist_controls.execution import execute, load_publication_inputs, dependency_context
from src.mnist_controls.aggregate import aggregate
from src.mnist_controls.artifacts import inspect_job, read_json


class FakeModel:
    dim=784
    device=torch.device('cpu')
    schedule=SimpleNamespace(T=1000)
    def __init__(self): self.times=[]; self.calls=[]
    def score(self,x,t):
        self.times.extend(torch.as_tensor(t).flatten().tolist())
        return -x
    def sample_ddim(self,n,num_steps,initial_noise):
        self.calls.append((n,num_steps,initial_noise.clone()))
        return initial_noise.tanh()


def fake_mnist(root,train,download):
    n=20 if train else 5
    return SimpleNamespace(data=torch.randint(0,256,(n,28,28),dtype=torch.uint8,
        generator=torch.Generator().manual_seed(91 if train else 92)),targets=torch.arange(n)%10)


def fixture(root):
    c=load_config()
    c.update(protocol='mnist_controls_tiny_test_v1',population_size=2,pool_size=4,batch_size=2,
             output_root=str(root/'output'),checkpoint=str(root/'source.pt'),device='cpu',
             publication_root=str(root/'no_old_outputs'),publication_suites={})
    c['evaluation'].update(evaluator_checkpoint=str(root/'evaluator.pt'),data_root=str(root/'data'),
                           reference_cache=str(root/'cache'),validation_limit=4,train_limit=8,knn_k=2)
    raw=Path(c['evaluation']['data_root'])/'MNIST/raw';raw.mkdir(parents=True)
    for name in ('train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte'):
        (raw/name).write_bytes(b'test fixture; MNIST reader is patched')
    torch.manual_seed(42)
    model=MNISTDiffusion(T=1000,base_channels=8,time_dim=16,device='cpu')
    state=dict(format='mnist_diffusion_v1',model_config=model.config,model=model.net.state_dict(),
        config=dict(seed=42),split_indices=dict(train=torch.arange(16),validation=torch.arange(16,20)))
    save_pt(c['checkpoint'],state)
    save_pt(c['evaluation']['evaluator_checkpoint'],dict(format='mnist_evaluator_v1',model=EvaluatorCNN().state_dict(),
        metadata=dict(train_indices=torch.arange(16),validation_indices=torch.arange(16,20))))
    return c


class ControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_01_ddim_nfe_and_output_count(self):
        fake=FakeModel(); counted=CountedModel(fake,128)
        out=ddim_105(counted,gaussian_noise(128,784,0))
        self.assertEqual(out.shape,(128,784)); self.assertEqual(counted.nfe,13440)
        self.assertEqual([n for n,s,z in fake.calls],[128]);self.assertEqual(fake.calls[0][1],105)
        self.assertEqual(ddim_indices(),torch.linspace(999,0,105).round().long().tolist())
        self.assertEqual(len(set(ddim_indices())),105)
        with patch('src.mnist_controls.controls.torch.linspace',return_value=torch.zeros(105)):
            with self.assertRaisesRegex(ValueError,'Duplicate'): ddim_indices()

    def test_02_real_sampler_counts_105_epsilon_calls(self):
        model=MNISTDiffusion(base_channels=8,time_dim=16,device='cpu')
        observed=[]
        handle=model.net.register_forward_hook(lambda m,args,out: observed.append((len(args[0]),int(args[1][0]))))
        counted=CountedModel(model,2)
        result=ddim_105(counted,gaussian_noise(2,784,2))
        handle.remove()
        self.assertEqual([t for n,t in observed],ddim_indices())
        self.assertEqual(sum(n for n,t in observed),210)
        self.assertEqual(result.shape,(2,784));self.assertTrue(torch.isfinite(result).all())

    def test_03_low_noise_nfe_only_t0_and_no_early_stop(self):
        fake=FakeModel(); counted=CountedModel(fake)
        initial=torch.zeros(128,784)
        out=low_noise_only(counted,initial)
        self.assertEqual(counted.nfe,7040);self.assertEqual(6400+counted.nfe,13440)
        self.assertEqual(set(fake.times),{0});self.assertEqual(len(fake.times),7040)
        self.assertTrue(torch.equal(initial,out));self.assertEqual(fake.calls,[])

    def test_04_pairing_and_prefix(self):
        z=gaussian_noise(268,784,1)
        self.assertTrue(torch.equal(z[:128],gaussian_noise(128,784,1)))
        fake=FakeModel();counted=CountedModel(fake)
        initial=counted.ddim(z[:128],50)
        pool=torch.cat((initial,counted.ddim(z[128:],50)))
        self.assertTrue(torch.equal(pool[:128],initial))
        original=initial.clone();low_noise_only(counted,initial)
        self.assertTrue(torch.equal(initial,original))
        ddim_105(counted,z[:128])
        self.assertTrue(torch.equal(fake.calls[-1][2],z[:128]))
        self.assertEqual([x[0] for x in fake.calls],[128,128,12,128])

    def test_05_selector_exact_original_indices_and_api(self):
        pool=gaussian_noise(268,784,2).tanh()
        out,ids,stats=select_representatives(pool)
        self.assertEqual(len(ids.unique()),128);self.assertTrue(torch.equal(out,pool[ids]))
        self.assertEqual(stats['selector_neural_nfe'],0)
        self.assertEqual(list(inspect.signature(select_representatives).parameters),['pool','count'])
        self.assertTrue(all(a>=b-1e-9 for a,b in zip(stats['objective_history'],stats['objective_history'][1:])))
        with self.assertRaises(TypeError):select_representatives(pool,evaluator=object())

    def test_06_selector_tie_break(self):
        _,ids,_=select_representatives(torch.zeros(268,784))
        self.assertEqual(ids.tolist(),list(range(128)))
        pool=torch.tensor([[-.8],[-.2],[.1],[.9]])
        _,ids,_=select_representatives(pool,3)
        # Independent brute force objective, including tie-breaking.
        d=(pool-pool.T).abs().numpy(); selected=[int(d.sum(0).argmin())]
        while len(selected)<3:
            candidates=[(float(d[:,selected+[j]].min(1).sum()),j) for j in range(4) if j not in selected]
            selected.append(min(candidates)[1])
        self.assertEqual(ids.tolist(),selected)

    def test_07_stability_identity(self):
        result,_=contraction_diagnostic(torch.zeros(5,784),lambda x:x,7)
        for r in result['summary']:
            self.assertAlmostEqual(r['median_rho'],1.,places=5)
            self.assertEqual(r['valid_fraction'],1.)
            self.assertEqual(r['median_reference_drift'],0.)

    def test_08_stability_known_contraction(self):
        result,_=contraction_diagnostic(torch.zeros(5,784),lambda x:.5*x,7)
        for r in result['summary']:self.assertAlmostEqual(r['mean_rho'],.5,places=5)

    def test_09_translation_drift_is_removed(self):
        result,_=contraction_diagnostic(torch.zeros(5,784),lambda x:x+.2,7)
        for r in result['summary']:
            self.assertAlmostEqual(r['mean_rho'],1.,places=5)
            self.assertAlmostEqual(r['median_reference_drift'],.2,places=5)

    def test_10_clipping_invalid_and_actual_denominator(self):
        x=torch.ones(4,784)
        with patch('src.mnist_controls.controls.torch.randn',return_value=torch.ones(4,2,784)):
            result,_=contraction_diagnostic(x,lambda x:x,1)
        for r in result['summary']:
            self.assertEqual(r['valid_fraction'],0.);self.assertIsNone(r['median_rho'])
        self.assertTrue(all(r['rho'] is None and r['delta0']==0 for r in result['per_candidate']))
        result,_=contraction_diagnostic(x,lambda x:x.clamp(-1,1),7)
        self.assertTrue(all(abs(r['rho']-1)<1e-5 for r in result['per_candidate'] if r['valid']))
        self.assertTrue(any(r['delta0']<r['scale'] for r in result['per_candidate']))

    def test_11_stability_nfe(self):
        counted=CountedModel(FakeModel())
        stability_v2(counted,torch.zeros(128,784),0)
        self.assertEqual(counted.nfe,6400)

    def test_12_plan_counts_budget_no_weights(self):
        c=load_config()
        # Never discover the user's real publication runs from a unit test.
        # These are presence-only fixtures, deliberately not loadable tensors.
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            c['publication_root']=str(root)
            c['publication_suites']={}
            with patch('torch.load',side_effect=AssertionError('dry run must not load tensors')):
                for available in range(4):
                    with self.subTest(complete_bundles=available):
                        if available:
                            seed=available-1
                            folder=root/f'seed{seed}';folder.mkdir()
                            (folder/'resolved_config.yaml').write_text(yaml.safe_dump(dict(
                                model_seed=42,population_size=128,search_seed=seed)))
                            (folder/'metadata.json').write_text('{}')
                            for method in ('rms_v3f2','matched_ddim'):
                                target=folder/method;target.mkdir()
                                for name in ('arrays.pt','summary.json','complete.json'):
                                    (target/name).write_bytes(b'presence-only test fixture')
                        p=plan(c)
                        self.assertEqual((p['quality_jobs'],p['stability_jobs']),(9,6))
                        self.assertEqual(p['fair_controls_and_diagnostic_nfe'],159240)
                        self.assertEqual(p['new_job_nfe'],99840)
                        self.assertEqual(p['dependency_nfe_upper'],(3-available)*20440)
                        self.assertEqual(p['total_new_nfe_upper'],99840+(3-available)*20440)
                        self.assertEqual({r['search_seed'] for r in p['jobs']},{0,1,2})
                        self.assertEqual(sum(r['complete_tensor_bundle'] for r in p['publication_sources'].values()),available)
                # An incomplete old bundle requires regenerating that seed's inputs.
                (root/'seed1/matched_ddim/arrays.pt').unlink()
                p=plan(c)
                self.assertEqual(p['dependency_nfe_upper'],20440)
                self.assertEqual(p['total_new_nfe_upper'],120280)

    @patch('src.mnist_experiments.data.MNIST',side_effect=fake_mnist)
    def test_13_end_to_end_tiny_real_model_resume_failures(self,_):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);c=fixture(root)
            # Only one control fails; subsequent jobs/seeds still execute and aggregate.
            original=select_representatives
            calls=[]
            def fail_once(pool,count):
                calls.append(1)
                if len(calls)==1:raise RuntimeError('injected selector failure')
                return original(pool,count)
            with patch('src.mnist_controls.execution.select_representatives',side_effect=fail_once):
                with self.assertRaisesRegex(RuntimeError,'failed/incompatible'):execute(c)
            out=Path(c['output_root']); report=aggregate(out)
            self.assertFalse(report['complete'])
            status=read_json(out/'summary/job_status.json')
            failed=[r for r in status if r['status']=='failed']
            self.assertEqual(len(failed),1);self.assertEqual(failed[0]['search_seed'],0)
            raw=read_json(out/'summary/raw_runs.json')
            self.assertEqual({r['search_seed'] for r in raw},{0,1,2})
            self.assertEqual(len(raw),18)
            finished=out/'model42_seed1/ddim_105/complete.json';stamp=finished.stat().st_mtime_ns
            self.assertTrue(execute(c,resume=True)['complete'])
            self.assertEqual(stamp,finished.stat().st_mtime_ns)
            with patch('src.mnist_controls.execution.load_backend',side_effect=AssertionError('must skip')):
                self.assertTrue(execute(c,resume=True)['complete'])
            with self.assertRaises(FileExistsError): execute(c)
            bad=deepcopy(c);bad['evaluation']['knn_k']=1
            with self.assertRaisesRegex(ValueError,'context changed'):execute(bad,resume=True)
            cache=next((root/'cache').glob('*.pt'))
            # Per-job corruption must be visible to aggregation and blocked on resume.
            target=out/'model42_seed2/low_noise_only/metrics.json'
            target.write_text('{}')
            report=aggregate(out);self.assertFalse(report['complete'])
            self.assertTrue(any(x['status']=='corrupt' for x in report['failed_or_missing']))
            with self.assertRaisesRegex(RuntimeError,'failed/incompatible'):execute(c,resume=True)
            state=torch.load(c['checkpoint'],weights_only=True);state['extra']='changed';save_pt(c['checkpoint'],state)
            with self.assertRaisesRegex(ValueError,'context changed'):execute(c,resume=True)

    @patch('src.mnist_experiments.data.MNIST',side_effect=fake_mnist)
    def test_14_import_validation_and_reference_cache_reuse(self,_):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);c=fixture(root); c['search_seeds']=[0]
            execute(c)
            out=Path(c['output_root']); data=torch.load(out/'model42_seed0/inputs/arrays.pt',weights_only=True)
            prov=read_json(out/'model42_seed0/inputs/metadata.json')
            baselines=read_json(out/'model42_seed0/inputs/summary.json')['baseline_rows']
            old=root/'old';old.mkdir()
            cfg=dict(model_seed=42,search_seed=0,population_size=2,initial_noise_seed=0,batch_size=2,ddim_steps=50,evaluation=c['evaluation'])
            (old/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
            metadata=dict(signature='legacy-test-signature',definition_hash=definition_hash(),implementation_sha256=code_digest(),
                checkpoint_sha256=prov['checkpoint_sha256'],evaluator_sha256=prov['evaluator_sha256'])
            (old/'metadata.json').write_text(json.dumps(metadata))
            for method in ('matched_ddim','rms_v3f2'):
                p=old/method;p.mkdir()
                row=next(r for r in baselines if r['method']==method)
                row.update(model_seed=42,search_seed=0,population_size=2,initial_noise_seed=0,
                           definition_hash=definition_hash(),resolved_method_definition=definitions()[method])
                (p/'summary.json').write_text(json.dumps(row));(p/'complete.json').write_text(json.dumps(dict(signature='legacy-test-signature',summary=row)))
                initial=data['pool'] if method=='matched_ddim' else data['initial']
                final=data['pool'] if method=='matched_ddim' else data['full']
                save_pt(p/'arrays.pt',dict(initial_noise=data['noise'][:len(initial)],initial=initial,candidates=final))
            cache=next((root/'cache').glob('*.pt'));stamp=cache.stat().st_mtime_ns
            c['output_root']=str(root/'reused');c['publication_suites']={0:str(old)}
            execute(c)
            self.assertEqual(stamp,cache.stat().st_mtime_ns)
            summary=read_json(root/'reused/model42_seed0/inputs/summary.json')
            self.assertEqual(summary['newly_executed_nfe'],0)
            old_cache=cache.read_bytes();cache.write_bytes(old_cache+b'corrupt')
            with self.assertRaisesRegex(ValueError,'Reference cache changed'):execute(c,resume=True)
            cache.write_bytes(old_cache)
            np.testing.assert_equal([r[k] for r in summary['baseline_rows'] for k in ('pixel_knn_mean','feature_knn_mean')],
                                    [r[k] for r in baselines for k in ('pixel_knn_mean','feature_knn_mean')])
            a=torch.load(old/'rms_v3f2/arrays.pt',weights_only=True);a['initial'][0,0]+=.01;save_pt(old/'rms_v3f2/arrays.pt',a)
            with self.assertRaises(ValueError):load_publication_inputs(old,c,dependency_context(c),0,data['noise'])

    def test_15_backend_failure_retains_expected_inventory(self):
        with TemporaryDirectory() as tmp:
            c=fixture(Path(tmp))
            with patch('src.mnist_controls.execution.load_backend',side_effect=ValueError('bad split')):
                with self.assertRaisesRegex(ValueError,'bad split'):execute(c)
            status=read_json(Path(c['output_root'])/'summary/job_status.json')
            self.assertEqual(len(status),18)
            self.assertTrue(all(r['status']=='failed' for r in status))
            self.assertEqual({r['search_seed'] for r in status},{0,1,2})

if __name__=='__main__':unittest.main()
