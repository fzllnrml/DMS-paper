"""Cheap frozen-protocol, orchestration and artifact tests. No MNIST downloads."""
from copy import deepcopy
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
from src.mnist_experiments.common import save_pt, merge_dict
from src.mnist_experiments.publication import (definitions, configure, definition_hash, matched_budget,
    METHODS, ABLATIONS, PROTOCOL)
from src.mnist_experiments.publication_evaluation import memorization_metrics, PublicationSupport
from src.mnist_experiments.publication_training import seed_training_config, train_job, load_training_source
from src.mnist_experiments.publication_aggregate import summarize, aggregate
from src.mnist_experiments.runner import expand_matrix, estimate, execute_jobs
from src.mnist_experiments.run import run_suite
from src.mnist_experiments.geometry import step_delta
from src.mnist_experiments.evaluator import EvaluatorCNN


def fake_mnist(root,train,download):
    n=20 if train else 5
    rng=torch.Generator().manual_seed(200 if train else 201)
    return SimpleNamespace(data=torch.randint(0,256,(n,28,28),generator=rng,dtype=torch.uint8),targets=torch.arange(n)%10)


def source(path):
    torch.manual_seed(42)
    model=MNISTDiffusion(T=1000,base_channels=8,time_dim=16,device='cpu')
    config=dict(seed=42,device='cpu',output_dir='runs/mnist',model=model.config,
        data=dict(root='fake',download=False,validation_size=4,split_seed=42,train_limit=None,validation_limit=None),
        training=dict(epochs=2,batch_size=4,lr=.0002,weight_decay=0.,grad_clip=1.,num_workers=0))
    state=dict(format='mnist_diffusion_v1',model_config=model.config,model=model.net.state_dict(),config=config,
               epoch=1,split_indices={'train':torch.arange(16),'validation':torch.arange(16,20),'test_size':5})
    save_pt(path,state)
    return state


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_fixed_definition_and_reject_tuning(self):
        d=definitions()
        c=d['rms_v3f2']
        self.assertEqual(c['schedule']['timesteps'],[800,500,300,200,100,50,20,5,0])
        self.assertEqual((c['alpha'],c['steps_per_level'],c['final_steps'],c['final_step_scale']),(.03,5,10,.1))
        self.assertEqual(c['merge']['schedule'],'none')
        self.assertEqual(d['original_v3f2'],merge_dict(c,{'step_mode':'l2'}))
        self.assertEqual(d['rms_v3f2_merge']['merge']['radius'],.65)
        x=torch.ones(4,784)
        torch.testing.assert_close(step_delta(x,.03,'rms'),torch.full_like(x,.03))
        self.assertEqual(definition_hash(),definition_hash())
        with self.assertRaises(ValueError): configure({'search':{'alpha':.04}})
        with self.assertRaises(ValueError): configure({'ddim_steps':51})
        with self.assertRaises(ValueError): configure({'methods':['matched_ddim']})

    def test_matrix_sizes_paths_costs(self):
        expected={'train':(2,0,0),'main':(30,1798200,733500),'ablation':(3,220800,0),'population':(12,604200,0)}
        for series,(n,search,diagnostic) in expected.items():
            jobs=expand_matrix(f'configs/mnist_experiments/publication_{series}.yaml')
            costs=[estimate(j) for j in jobs]
            self.assertEqual(len(jobs),n)
            self.assertEqual(sum(e.get('nfe_upper_bound') or 0 for e in costs),search)
            self.assertEqual(sum(e.get('diagnostic_nfe_upper',0) for e in costs),diagnostic)
            if series=='train':
                self.assertEqual(sum(e['model_trainings'] for e in costs),2)
                self.assertEqual([j['config']['training_output'] for j in jobs],['runs/mnist_seed43','runs/mnist_seed44'])
            else:
                for j in jobs:
                    self.assertEqual(j['config']['search']['alpha'],.03)
                    self.assertEqual(j['config']['initial_noise_seed'],j['config']['search_seed'])
        main=expand_matrix('configs/mnist_experiments/publication_main.yaml')
        self.assertEqual({j['config']['checkpoint'] for j in main},{'runs/mnist/checkpoints/best.pt','runs/mnist_seed43/checkpoints/best.pt','runs/mnist_seed44/checkpoints/best.pt'})
        self.assertTrue(all('pca' not in str(j) for j in main))

    def test_matched_calculation(self):
        self.assertEqual(matched_budget(13440),dict(target_v3f2_nfe=13440,matched_ddim_sample_count=268,matched_ddim_nfe=13400,nfe_difference=-40))
        for nfe in (6720,13440,26880,101):
            b=matched_budget(nfe)
            self.assertTrue(-50 < b['nfe_difference'] <= 0)
        with self.assertRaises(ValueError): matched_budget(49)

    def test_memorization_reference_calibration(self):
        m=memorization_metrics([0.,.5],[.4,.6],[.2,.3,.4],[.3,.4,.5])
        self.assertEqual(m['summary']['fraction_unusually_close_train'],.5)
        self.assertAlmostEqual(m['summary']['mean_nearest_train_distance'],.25)
        self.assertEqual(len(m['gap_percentiles']),2)
        with self.assertRaises(ValueError): memorization_metrics([1],[1,2],[1],[2])

    def test_seed_training_recipe(self):
        with TemporaryDirectory() as tmp:
            state=source(Path(tmp)/'source.pt')
            for seed in (43,44):
                cfg=seed_training_config(state,seed,f'runs/mnist_seed{seed}','mps')
                for key in ('model','training','data'): self.assertEqual(cfg[key],state['config'][key])
                self.assertEqual(cfg['seed'],seed)
            bad=deepcopy(state);bad['config']['data']['split_seed']=None
            # Missing fixed split seed must be refused.
            bad['config']['data'].pop('split_seed')
            with self.assertRaises(ValueError): seed_training_config(bad,43,'out','cpu')

    def test_seed42_completed_training_recipe(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);best=source(root/'best.pt');latest=deepcopy(best)
            latest['config']['training']['epochs']=3;latest['epoch']=3
            save_pt(root/'latest.pt',latest)
            c=dict(source_checkpoint=str(root/'best.pt'),source_latest_checkpoint=str(root/'latest.pt'))
            self.assertEqual(load_training_source(c)['config']['training']['epochs'],3)
            latest['epoch']=2;save_pt(root/'latest.pt',latest)
            with self.assertRaises(ValueError): load_training_source(c)

    def test_training_driver_interrupt_resume_skip(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);state=source(root/'source.pt')
            job=dict(output=str(root/'job'),config=dict(kind='publication_train',source_checkpoint=str(root/'source.pt'),
                model_seed=43,training_output=str(root/'train'),device='cpu'))
            calls=[]
            def trainer(command,check):
                cfg=yaml.safe_load(Path(command[command.index('--config')+1]).read_text())
                calls.append(command)
                new=deepcopy(state);new['config']=cfg;new['epoch']=1 if len(calls)==1 else 2
                target=Path(cfg['output_dir'])/'checkpoints'
                save_pt(target/'best.pt',new);save_pt(target/'latest.pt',new)
                if len(calls)==1: raise RuntimeError('simulated interruption after one completed epoch')
            spec=SimpleNamespace(origin=str(Path('src/mnist/model.py').resolve()))
            with patch('src.mnist_experiments.publication_training.importlib.util.find_spec',return_value=spec),patch('src.mnist_experiments.publication_training.verify_split'),patch('src.mnist_experiments.publication_training.subprocess',SimpleNamespace(run=trainer)):
                with self.assertRaises(RuntimeError): train_job(job)
                train_job(job,resume=True)
                self.assertIn('--resume',calls[1])
                stamp=(root/'train/checkpoints/best.pt').stat().st_mtime_ns
                train_job(job,resume=True)
                self.assertEqual(len(calls),2)
                self.assertEqual(stamp,(root/'train/checkpoints/best.pt').stat().st_mtime_ns)
                with self.assertRaises(FileExistsError): train_job(job,resume=False)
                changed=deepcopy(job);changed['config']['model_seed']=44
                with self.assertRaises(ValueError): train_job(changed,resume=True)

    def test_bootstrap_paired_validation(self):
        rows=[]
        for seed in [42,43,44]:
            for search in range(3):
                for method,value in [('rms_v3f2',1.),('matched_ddim',3.)]:
                    rows.append(dict(publication_series='test',population_size=128,definition_hash='d',evaluation_definition_hash='e',
                        evaluator_sha256='ev',model_seed=seed,search_seed=search,initial_noise_seed=search,method=method,
                        checkpoint_sha256=str(seed),initial_gaussian_prefix_sha256=str(search),initial_ddim_prefix_sha256=str(search),
                        pixel_knn_distance=value))
        summary,paired=summarize(rows,50,7)
        self.assertEqual(paired[0]['mean'],-2.)
        self.assertEqual(paired[0]['median'],-2.)
        self.assertEqual((paired[0]['ci95_low'],paired[0]['ci95_high']),(-2.,-2.))
        self.assertEqual(paired[0]['n_runs'],9)
        self.assertEqual((summary,paired),summarize(rows,50,7))
        bad=deepcopy(rows);bad[0]['initial_noise_seed']=99
        with self.assertRaises(ValueError): summarize(bad,50)
        with self.assertRaises(ValueError): summarize(rows+[rows[0]],50)

    @patch('src.mnist_experiments.data.MNIST',side_effect=fake_mnist)
    def test_full_publication_smoke_resume_cache_aggregation(self,_):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);source(root/'source.pt')
            save_pt(root/'evaluator.pt',dict(format='mnist_evaluator_v1',model=EvaluatorCNN().state_dict(),
                metadata=dict(train_indices=torch.arange(16),validation_indices=torch.arange(16,20),validation_accuracy=0.)))
            cfg=configure(dict(kind='publication_search',publication_series='smoke',checkpoint=str(root/'source.pt'),device='cpu',
                model_seed=42,search_seed=0,population_size=2,methods=METHODS+ABLATIONS,batch_size=2,
                evaluation=dict(evaluator_checkpoint=str(root/'evaluator.pt'),require_evaluator=True,train_limit=8,validation_limit=4,
                                knn_k=2,include_test=True,reference_cache=str(root/'cache')),
                stability=dict(enabled=True,directions=1,steps=1,scales=[.01]),
                visualization=dict(enabled=True,max_images=2,trajectory_particles=0)))
            rows=run_suite(cfg,root/'suite')
            by_method={r['method']:r for r in rows}
            self.assertEqual(len(rows),9)
            self.assertEqual({r['initial_gaussian_prefix_sha256'] for r in rows}, {rows[0]['initial_gaussian_prefix_sha256']})
            self.assertEqual({r['initial_ddim_prefix_sha256'] for r in rows}, {rows[0]['initial_ddim_prefix_sha256']})
            for method in ('matched_ddim','matched_cluster'):
                r=by_method[method]
                self.assertEqual(r['target_v3f2_nfe'],by_method[r['matched_target_method']]['nfe'])
                self.assertEqual(r['nfe'],r['matched_ddim_nfe'])
                self.assertEqual(r['nfe_difference'],r['nfe']-r['target_v3f2_nfe'])
            self.assertEqual(by_method['rms_v3f2']['nfe'],210)
            self.assertEqual(by_method['rms_v3f2_merge']['nfe'],210)
            self.assertGreater(by_method['rms_v3f2']['diagnostic_nfe'],0)
            self.assertEqual(by_method['original_v3f2']['diagnostic_nfe'],0)
            self.assertIn('fraction_unusually_close_train',rows[0])
            self.assertTrue((root/'suite/paired_grid.png').exists())
            cache=next((root/'cache').glob('*.pt'));stamp=cache.stat().st_mtime_ns
            with patch('src.mnist_experiments.run.load_model',side_effect=AssertionError('must skip')):
                self.assertEqual(rows,run_suite(cfg,root/'suite',resume=True))
            original_stamp=(root/'suite/rms_v3f2/complete.json').stat().st_mtime_ns
            (root/'suite/complete.json').unlink();(root/'suite/matched_cluster/complete.json').unlink()
            run_suite(cfg,root/'suite',resume=True)
            self.assertEqual(stamp,cache.stat().st_mtime_ns)
            self.assertEqual(original_stamp,(root/'suite/rms_v3f2/complete.json').stat().st_mtime_ns)
            aggregate([root/'suite'],root/'summary',50,plots=True)
            for name in ['raw_runs.csv','summary.csv','paired_comparisons.csv','publication_summary.md']:
                self.assertTrue((root/'summary'/name).exists())
            self.assertTrue(list((root/'summary/plots').rglob('*.pdf')))
            bad=deepcopy(cfg);bad['evaluation']['knn_k']=1
            metadata=(root/'suite/metadata.json').read_bytes()
            with self.assertRaises(ValueError): execute_jobs([dict(id='bad',output=str(root/'suite'),config=bad)],resume=True)
            self.assertEqual(metadata,(root/'suite/metadata.json').read_bytes())

    def test_legacy_feature_distance_aggregation(self):
        from src.mnist_experiments.aggregate import summarize as legacy
        row=dict(suite_id='s',condition_id='c',variant='v',method='original',space='pixel',dimension=784,
                 model_seed=42,search_seed=0,initial_noise_seed=0,feature_knn_mean=2.)
        summary,_=legacy([row],50)
        self.assertEqual(summary[0]['metric'],'feature_knn_mean')

if __name__=='__main__': unittest.main()
