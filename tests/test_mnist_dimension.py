"""Offline scientific-contract tests and tiny real-code training/search smoke."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json,shutil,unittest
from unittest.mock import patch
import numpy as np
import torch
from src.mnist import MNISTDiffusion
from src.mnist_experiments.common import save_pt,CountedModel
from src.mnist_experiments.publication import definitions
from src.mnist_experiments.geometry import step_delta
from src.mnist_experiments.search import run_search
from src.models.noise_schedule import NoiseSchedule
from src.mnist_experiments.evaluator import EvaluatorCNN
from src.mnist_dimension.design import load_config,plan,models,method_config,METHODS
from src.mnist_dimension.representation import fit_basis,CachedPCASpace,prepare
from src.mnist_dimension.training import train_model
from src.mnist_dimension.execution import movement_summary,search_job,initialize
from src.mnist_dimension.evaluation import KNN
from src.mnist_dimension.runner import execute
from src.mnist_dimension.aggregate import aggregate,summarize


def fake_mnist(root,train,download):
    n=32 if train else 8
    data=torch.randint(0,256,(n,28,28),generator=torch.Generator().manual_seed(31 if train else 32),dtype=torch.uint8)
    return SimpleNamespace(data=data,targets=torch.arange(n)%10)


def fixture(root):
    model=MNISTDiffusion(T=1000,base_channels=8,time_dim=16,device='cpu')
    state=dict(format='mnist_diffusion_v1',model=model.net.state_dict(),model_config=model.config,epoch=2,
               config={'seed':42,'training':dict(epochs=2,batch_size=4,lr=.0002,weight_decay=0.,grad_clip=1.)},
               split_indices={'train':torch.arange(24),'validation':torch.arange(24,32),'test_size':8})
    save_pt(root/'source.pt',state)
    save_pt(root/'evaluator.pt',dict(format='mnist_evaluator_v1',model=EvaluatorCNN().state_dict(),
        metadata=dict(train_indices=torch.arange(24),validation_indices=torch.arange(24,32))))
    cfg=load_config('configs/mnist_dimension.yaml','cpu',smoke=True)
    cfg.update(root=str(root/'study'),source_checkpoint=str(root/'source.pt'),recipe_checkpoint=str(root/'source.pt'),
               evaluator_checkpoint=str(root/'evaluator.pt'),pixel_checkpoints={42:str(root/'source.pt')},
               data_root='unused',max_visual_images=2)
    return cfg


class DimensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_fixed_design_and_budget(self):
        cfg=load_config('configs/mnist_dimension.yaml');p=plan(cfg)
        self.assertEqual((p['representation_fits'],p['diffusion_trainings'],p['quality_jobs'],p['search_jobs']),(1,13,16,80))
        self.assertEqual((p['search_nfe_upper_bound'],p['diagnostic_nfe_upper_bound'],p['total_inference_nfe_upper_bound']),(3734400,131072,3865472))
        self.assertEqual(sum(j['representation']=='pca' for j in p['jobs']),65)
        self.assertFalse(any('publication_main' in j['checkpoint'] for j in p['jobs']))

    def test_pixel_definition_and_scaling_control(self):
        for method in METHODS:self.assertEqual(method_config(method,'pixel'),definitions()[method])
        for rep in ('pixel','pca'):
            a=method_config('original_v3f2',rep);b=method_config('rms_v3f2',rep)
            a['step_mode']='rms';self.assertEqual(a,b)
        for d in [16,32,64,128,256,512,784]:
            score=torch.arange(1,d+1).float()[None]
            l2=step_delta(score,.03,'l2');rms=step_delta(score,.03,'rms')
            torch.testing.assert_close(rms,step_delta(score,.03*np.sqrt(d),'l2'))
            self.assertAlmostEqual(float(l2.square().mean().sqrt()),.03/np.sqrt(d),places=7)
            self.assertAlmostEqual(float(rms.square().mean().sqrt()),.03,places=7)

    def test_train_only_fit_and_inverse(self):
        raw=fake_mnist('',True,False).data[:16]
        a=fit_basis(raw,[16,784]);b=fit_basis(raw.clone(),[16,784])
        torch.testing.assert_close(a['components'],b['components'],rtol=0,atol=0)
        full=CachedPCASpace(a,784);images=torch.randn(3,1,28,28)
        torch.testing.assert_close(full.decode(full.encode(images)),images,atol=3e-6,rtol=3e-6)
        small=CachedPCASpace(a,16);z=small.encode(images)
        torch.testing.assert_close(small.encode(small.decode(z)),z,atol=3e-6,rtol=3e-6)
        self.assertEqual(a['fit_n'],16);self.assertFalse(a['whiten'])

    @patch('src.mnist_experiments.data.MNIST',side_effect=fake_mnist)
    def test_heldout_images_and_labels_cannot_change_fit(self,_):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=fixture(root);prepare(cfg)
            first=torch.load(Path(cfg['root'])/'representation/basis.pt',weights_only=True)
            other=deepcopy(cfg);other['root']=str(root/'changed_heldout')
            def changed(root,train,download):
                data=fake_mnist(root,train,download)
                if train:data.data[24:]=255-data.data[24:]
                data.targets=9-data.targets
                return data
            with patch('src.mnist_experiments.data.MNIST',side_effect=changed):prepare(other)
            second=torch.load(Path(other['root'])/'representation/basis.pt',weights_only=True)
            torch.testing.assert_close(first['components'],second['components'],rtol=0,atol=0)
            torch.testing.assert_close(first['mean'],second['mean'],rtol=0,atol=0)
            self.assertEqual(first['fit_pixels_sha256'],second['fit_pixels_sha256'])

    def test_knn_exhaustive_near_copy(self):
        ref=torch.randn(70,16,generator=torch.Generator().manual_seed(3));q=torch.cat((ref[:2],ref[:2]+1e-6,torch.ones(1,16)))
        distances,indices=KNN(ref,5).query(q)
        expected=torch.cdist(q.double(),ref.double(),compute_mode='donot_use_mm_for_euclid_dist').topk(5,largest=False).values.float()
        torch.testing.assert_close(distances,expected,atol=1e-7,rtol=1e-6)
        self.assertEqual(distances[0,0],0)

    def test_measured_movement_and_decoded_scale(self):
        for d in (16,784):
            state=dict(components=torch.eye(784),mean=torch.zeros(784));space=CachedPCASpace(state,d)
            model=SimpleNamespace(dim=d,device=torch.device('cpu'),schedule=NoiseSchedule.linear(1000,.0001,.02,torch.device('cpu')),score=lambda x,t:-x)
            for method in ('original_v3f2','rms_v3f2'):
                c=method_config(method,'pca');initial=torch.full((2,d),10.)
                found=run_search(CountedModel(model,2),initial,c,space)
                row=movement_summary(found,initial,space,c['step_mode'])
                expected=.03/(np.sqrt(d) if method=='original_v3f2' else 1)
                self.assertAlmostEqual(row['regular_actual_step_rms'],expected,places=5)
                self.assertAlmostEqual(row['regular_actual_pixel_step_rms'],expected*np.sqrt(d/784),places=5)
                self.assertGreater(row['cumulative_rms_path'],0)

    @patch('src.mnist_experiments.data.MNIST',side_effect=fake_mnist)
    def test_actual_training_resume_exact(self,_):
        from src.mnist_dimension import training
        with TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=fixture(root);cfg['epochs_override']=2
            prepare(cfg);spec=models(cfg)[0];real_save=training.save_pt
            def interrupt(path,value):
                real_save(path,value)
                if Path(path).name=='latest.pt' and value['epoch']==1:raise RuntimeError('simulated epoch boundary interruption')
            with patch.object(training,'save_pt',side_effect=interrupt):
                with self.assertRaises(RuntimeError):train_model(cfg,spec)
            train_model(cfg,spec,resume=True)
            resumed=torch.load(spec['checkpoint'],weights_only=True)
            other=deepcopy(cfg);other['root']=str(root/'continuous')
            shutil.copytree(Path(cfg['root'])/'representation',Path(other['root'])/'representation')
            continuous_spec=models(other)[0];train_model(other,continuous_spec)
            uninterrupted=torch.load(continuous_spec['checkpoint'],weights_only=True)
            for k in resumed['model']:torch.testing.assert_close(resumed['model'][k],uninterrupted['model'][k],rtol=0,atol=0)
            self.assertEqual(resumed['best_loss'],uninterrupted['best_loss'])
            with patch.object(training,'recipe',side_effect=AssertionError('completed training should skip')):
                train_model(cfg,spec,resume=True)

    def test_paired_bootstrap_and_strata(self):
        rows=[]
        for d in (16,32):
            for model in ([42,43,44] if d==16 else [42]):
                for seed in [0,1,2]:
                    for method,value in [('rms_v3f2',1.),('original_v3f2',2.),('matched_ddim',3.)]:
                        rows.append(dict(representation='pca',dimension=d,model_seed=model,search_seed=seed,method=method,
                            checkpoint_sha256=str(model),representation_sha256='rep',initial_noise_seed=seed,
                            initial_noise_prefix_sha256=str(seed),initial_ddim_prefix_sha256=str(seed),evaluation_key='e',
                            feature_knn_distance=value))
        a,b=summarize(rows,[16],50)
        curve=[r for r in a if r['stratum']=='curve_seed42'];anchors=[r for r in a if r['stratum']=='replicated_anchors']
        self.assertTrue(all(r['n_model_seeds']==1 for r in curve));self.assertTrue(all(r['n_model_seeds']==3 for r in anchors))
        self.assertTrue(all(r['ci95_low']==r['mean']==r['ci95_high'] for r in b))
        self.assertEqual((a,b),summarize(rows,[16],50))
        bad=deepcopy(rows);bad[0]['initial_noise_seed']=100
        with self.assertRaises(ValueError):summarize(bad,[16],50)
        with self.assertRaises(ValueError):summarize(rows+[rows[0]],[16],50)

    @patch('src.mnist_experiments.data.MNIST',side_effect=fake_mnist)
    def test_end_to_end_smoke_quality_failure_visibility_and_resume(self,_):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=fixture(root)
            execute(cfg,'train');execute(cfg,'search')
            expected=plan(cfg);summary,paired,status=aggregate(cfg['root'],50,plots=True)
            self.assertEqual(sum(s['kind']=='search' and s['status']=='completed' for s in status),3)
            for j in expected['jobs']:
                rows=json.loads((Path(j['output'])/'summary.json').read_text());by={r['method']:r for r in rows}
                self.assertEqual(len({r['initial_noise_prefix_sha256'] for r in rows}),1)
                self.assertEqual(len({r['initial_ddim_prefix_sha256'] for r in rows}),1)
                self.assertEqual(by['matched_ddim']['target_v3f2_nfe'],by['rms_v3f2']['nfe'])
                self.assertTrue(-50<by['matched_ddim']['nfe_difference']<=0)
                self.assertTrue(all(r['diagnostic_nfe']==0 for r in rows))
            with patch('src.mnist_dimension.evaluation.StudyEvaluator',side_effect=AssertionError('complete resume should not load data')):
                execute(cfg,'search',resume=True)
            first=Path(expected['jobs'][0]['output']);stamp=(first/'rms_v3f2/summary.json').stat().st_mtime_ns
            (first/'complete.json').unlink();(first/'matched_ddim/complete.json').unlink()
            execute(cfg,'search',resume=True)
            self.assertEqual(stamp,(first/'rms_v3f2/summary.json').stat().st_mtime_ns)
            self.assertTrue(list((Path(cfg['root'])/'summary/plots').glob('*.pdf')))
            (first/'complete.json').unlink();(first/'original_v3f2/complete.json').unlink()
            with patch('src.mnist_dimension.execution.run_search',side_effect=FloatingPointError('nonfinite score fixture')):
                with self.assertRaises(RuntimeError):execute(cfg,'search',resume=True)
            _,_,failure_status=aggregate(cfg['root'],50,plots=False)
            self.assertTrue(any(s['status']=='failed' for s in failure_status))
            execute(cfg,'search',resume=True)

            # Corrupt completed method must be visible, not silently excluded as a good point.
            (first/'matched_ddim/arrays.pt').unlink()
            _,_,statuses=aggregate(cfg['root'],50,plots=False)
            self.assertTrue(any(s['status']=='corrupt_method_artifact' for s in statuses))
            table=json.loads((Path(cfg['root'])/'summary/scientific_table.json').read_text())
            self.assertTrue(any(r['completed_jobs']<r['expected_jobs'] for r in table))

if __name__=='__main__':unittest.main()
