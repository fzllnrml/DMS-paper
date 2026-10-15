"""Offline correctness and tiny end-to-end tests, with no scientific runs."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch,yaml
from src.mnist import MNISTDiffusion
from src.models.noise_schedule import NoiseSchedule
from src.mnist_experiments.common import CountedModel,PixelSpace,gaussian_noise,merge_dict,save_pt
from src.mnist_experiments.config import DEFAULT
from src.mnist_experiments.geometry import step_delta,project,distances,merge_particles,should_merge,radius_at
from src.mnist_experiments.search import resolve_levels,run_search
from src.mnist_experiments.evaluation import semantic_metrics,nearest_neighbors,support_percentiles,local_stability,directional_curvature
from src.mnist_experiments.runner import expand_matrix,estimate
from src.mnist_experiments.run import run_suite
from src.mnist_experiments.pca import PCASpace,PCADiffusion,fit_pca,train_pca
from src.mnist_experiments.evaluator import EvaluatorCNN,FrozenEvaluator,train_evaluator
from src.mnist_experiments.aggregate import summarize

class Gaussian:
    def __init__(self,dim=2):
        self.dim=dim; self.device=torch.device('cpu'); self.schedule=NoiseSchedule.linear(12,.0001,.02,self.device); self.nfe=0; self.calls=[]
    def score(self,x,t): self.calls.append(len(x)); self.nfe+=len(x); return -x
    def score_numpy(self,x,t): return self.score(torch.as_tensor(x).reshape(-1,self.dim),t).numpy().reshape(x.shape)
    def enable_nfe_counting(self): self.nfe=0
    def reset_nfe(self): self.nfe=0
FLAT=SimpleNamespace(name='pixel',decode=lambda x:x)
def fake_mnist(root,train,download):
    n=16 if train else 5
    return SimpleNamespace(data=(torch.arange(n*784).reshape(n,28,28)%256).to(torch.uint8),targets=torch.arange(n)%10)
def source_checkpoint(path):
    torch.manual_seed(41); model=MNISTDiffusion(T=12,base_channels=8,time_dim=16,device='cpu')
    save_pt(path,dict(format='mnist_diffusion_v1',model_config=model.config,model=model.net.state_dict(),config={'seed':42},split_indices={'train':torch.arange(12),'validation':torch.arange(12,16),'test_size':5}))
    return model

class ExperimentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.threads=torch.get_num_threads(); torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)
    def setUp(self): torch.manual_seed(1)
    def test_image_counted_batched_score(self):
        base=Gaussian(); model=CountedModel(base,3); x=torch.ones(8,2)
        torch.testing.assert_close(model.score(x,torch.arange(8)),-x)
        self.assertEqual((model.nfe,model.forward_calls,base.calls),(8,3,[3,3,2]))
    def test_step_scaling(self):
        s=torch.tensor([[3.,4.],[0.,0.]])
        self.assertAlmostEqual(step_delta(s,.2,'l2')[0].norm().item(),.2,places=6)
        self.assertAlmostEqual(step_delta(s,.2,'rms')[0].square().mean().sqrt().item(),.2,places=6)
        torch.testing.assert_close(step_delta(s,.2,'raw'),.2*s); self.assertEqual(step_delta(s,.2,'l2')[1].norm(),0)
    def test_all_distances(self):
        x=torch.tensor([[1.,0.],[0.,1.],[0.,0.]]); d=distances(x,'euclidean',FLAT)
        np.testing.assert_allclose(distances(x,'rms_euclidean',FLAT),d/np.sqrt(2))
        self.assertEqual(distances(x,'cosine',FLAT)[0,1],1); self.assertEqual(distances(x,'cosine',FLAT)[2,2],0)
        ev=SimpleNamespace(features=lambda x:2*x)
        np.testing.assert_allclose(distances(x,'feature_l2',FLAT,ev),2*d)
        np.testing.assert_allclose(distances(x,'feature_cosine',FLAT,ev),distances(x,'cosine',FLAT))
        with self.assertRaises(ValueError): distances(x,'feature_l2',FLAT)
    def test_complete_linkage_mean_medoid(self):
        c=deepcopy(DEFAULT['search']['merge']); x=torch.tensor([[0.,0.],[.9,0.],[1.8,0.]])
        merged,_,info=merge_particles(x,1.,c,FLAT); self.assertEqual(len(merged),2); self.assertEqual(sum(info['cluster_sizes']),3)
        x=torch.tensor([[-1.,0.],[0.,0.],[.8,0.]])
        mean,_,_=merge_particles(x,3,c,FLAT); c['representative']='medoid'; medoid,_,_=merge_particles(x,3,c,FLAT)
        torch.testing.assert_close(mean,x.mean(0,keepdim=True)); torch.testing.assert_close(medoid,x[1:2])
    def test_merge_schedules_projection_radius(self):
        self.assertTrue(should_merge('every_level',0,4,[])); self.assertFalse(should_merge('final_only',0,4,[]))
        self.assertTrue(should_merge('selected_levels',0,4,[4])); self.assertFalse(should_merge('none',0,0,[],True))
        self.assertTrue(should_merge('final_only',0,0,[],True))
        x=torch.tensor([[1.1,-1.2]]); y,stats=project(x,'clamp',FLAT)
        torch.testing.assert_close(y,torch.tensor([[1.,-1.]])); self.assertAlmostEqual(stats['mean_overshoot'],.15,places=6); self.assertEqual(stats['fraction_outside_range'],1)
        torch.testing.assert_close(project(x,'none',FLAT)[0],x)
        c=dict(DEFAULT['search']['merge'],noise_units='pairwise_sigma',min_radius=0.,coefficient=1.); model=Gaussian(784)
        self.assertAlmostEqual(radius_at(c,model,5),model.schedule.sigma(5)*np.sqrt(1568))
        c['metric']='feature_l2'
        with self.assertRaises(ValueError): radius_at(c,model,5)
    def test_time_schedules(self):
        model=Gaussian(); levels,meta=resolve_levels({'kind':'explicit','timesteps':[11,5,0]},model.schedule)
        self.assertEqual(levels,[11,5,0]); self.assertGreater(meta['sigma_at_zero'],0)
        for kind,values in [('sigma',[model.schedule.sigma(0),model.schedule.sigma(11)]),('log_snr',[float(torch.logit(model.schedule.alphas_cumprod[t])) for t in (0,11)])]:
            self.assertEqual(resolve_levels({'kind':kind,'values':values},model.schedule)[0],[11,0])
        with self.assertRaises(ValueError): resolve_levels({'kind':'explicit','timesteps':[0,5]},model.schedule)
    def test_original_matches_legacy(self):
        from src.algorithms.mode_finder_v3_f2 import ModeFinderV3F2
        initial=torch.tensor([[-.8,.2],[.4,.5],[.42,.51],[.9,-.2]])
        c=merge_dict(DEFAULT['search'],dict(schedule={'kind':'explicit','timesteps':[11,5,0]},alpha=.04,steps_per_level=2,final_steps=3))
        model=CountedModel(Gaussian()); found=run_search(model,initial,c,FLAT)
        expected=ModeFinderV3F2(Gaussian(),timesteps=[11,5,0],step_size=.04,ascent_steps=2,refine_steps=3,refine_step_scale=.1,merge_factor=.8,merge_radius_min=.05,x_min=-1,x_max=1).find_modes(starts=initial.numpy(),verbose=False)
        np.testing.assert_allclose(found['candidates'],expected.modes,atol=2e-6); self.assertEqual(model.nfe,expected.nfe)
        self.assertEqual(len(found['trajectory']),4); self.assertEqual(sorted(sum(found['lineage'],[])),list(range(4)))
    def test_ddim_pairing_and_checkpoint(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt'; model=source_checkpoint(path)
            x=torch.randn(3,1,28,28,generator=torch.Generator().manual_seed(17))
            torch.testing.assert_close(model.sample_ddim(n=3,num_steps=4,seed=17),model.sample_ddim(n=3,num_steps=4,initial_noise=x),rtol=0,atol=0)
            noise=gaussian_noise(5,784,7); torch.testing.assert_close(noise[:3],gaussian_noise(3,784,7))
            a=CountedModel(model,2); b=CountedModel(model,5)
            torch.testing.assert_close(a.ddim(noise,4),b.ddim(noise,4),rtol=2e-5,atol=2e-6); self.assertEqual((a.nfe,b.nfe),(20,20))
            loaded=MNISTDiffusion.from_checkpoint(path,'cpu'); torch.testing.assert_close(model.score(noise,0),loaded.score(noise,0))
    def test_stability_curvature(self):
        model=CountedModel(Gaussian()); x=torch.zeros(2,2)
        c=dict(seed=1,directions=3,scales=[.01,.02],steps=12,alpha=.5,step_mode='raw',projection='none',t=0,return_radius=.001)
        stability=local_stability(model,x,FLAT,c)
        self.assertTrue(all(torch.all(r['return_rate']==1) for r in stability['results']))
        curvature=directional_curvature(model,x,dict(seed=2,directions=5,epsilon=.01,t=0))
        torch.testing.assert_close(curvature['values'],-torch.ones(2,5),rtol=1e-6,atol=1e-6); self.assertEqual(curvature['nfe'],20)
    def test_semantics_knn_percentiles(self):
        train=torch.tensor([[0.,0.],[1.,0.],[2.,0.],[3.,0.]])
        d,idx=nearest_neighbors(torch.tensor([[.1,0.],[2.9,0.]]),train,2,1,2)
        torch.testing.assert_close(d,torch.tensor([[.1,.9],[.1,.9]]),rtol=1e-5,atol=1e-6); self.assertEqual(idx[:,0].tolist(),[0,3])
        np.testing.assert_allclose(support_percentiles([1.,3.],[1.,2.,3.])['distance_percentile'],[100/6,500/6])
        metrics=semantic_metrics(torch.eye(10)*10); self.assertEqual(metrics['class_coverage'],10); self.assertAlmostEqual(metrics['class_entropy'],np.log(10),places=5)
    def test_matrix_dry_run(self):
        with TemporaryDirectory() as directory:
            p=Path(directory)/'m.yaml'; p.write_text(yaml.safe_dump({'experiments':[{'name':'a','grid':{'model_seed':[42,43],'search_seed':[0,1,2]},'config':{'checkpoint':'model{model_seed}.pt'}}]}))
            jobs=expand_matrix(p); self.assertEqual(len(jobs),6); self.assertEqual(jobs[2]['config']['initial_noise_seed'],2); self.assertEqual(jobs[3]['config']['checkpoint'],'model43.pt')
        jobs=expand_matrix('configs/mnist_experiments/all.yaml'); self.assertEqual(len(jobs),155); self.assertEqual(sum(estimate(j)['model_trainings'] for j in jobs),10)
    def test_pca_transform_inverse(self):
        state={'components':torch.eye(784),'mean':torch.zeros(784)}; space=PCASpace(state,784); x=torch.randn(3,1,28,28)
        torch.testing.assert_close(space.decode(space.encode(x)),x)
        small=PCASpace(state,16); z=small.encode(x); torch.testing.assert_close(small.encode(small.decode(z)),z)
        with self.assertRaises(ValueError): project(z,'clamp',small)
    @patch('src.mnist_experiments.data.MNIST',side_effect=fake_mnist)
    def test_tiny_pca_train_resume(self,_):
        with TemporaryDirectory() as directory:
            root=Path(directory); source_checkpoint(root/'source.pt'); fit_pca(root/'source.pt',root/'pca.pt',[2,4])
            path=train_pca(root/'pca.pt',root/'mlp',4,device='cpu',epochs=1,batch_size=4,model_options={'T':12,'hidden':[8]})
            train_pca(root/'pca.pt',root/'mlp',4,device='cpu',epochs=2,batch_size=4,resume=True,model_options={'T':12,'hidden':[8]})
            cfg=merge_dict(DEFAULT,{'checkpoint':str(path),'device':'cpu','population_size':2,'ddim_steps':3,'methods':['original'],
                'search':{'schedule':{'kind':'explicit','timesteps':[5,0]},'steps_per_level':1,'final_steps':1,'projection':'none'},
                'evaluation':{'train_limit':8,'validation_limit':4,'knn_k':2},'visualization':{'enabled':False}})
            rows=run_suite(cfg,root/'pca_search'); self.assertEqual(rows[0]['dimension'],4); self.assertIsNotNone(rows[0]['validation_reconstruction_rms'])
            model=PCADiffusion.from_checkpoint(path,'cpu'); self.assertEqual(model.sample_ddim(n=2,num_steps=3).shape,(2,4)); self.assertTrue(torch.isfinite(model.score(torch.zeros(3,4),torch.tensor([0,5,11]))).all())
    @patch('src.mnist_experiments.data.MNIST',side_effect=fake_mnist)
    def test_tiny_evaluator_train_resume(self,_):
        with TemporaryDirectory() as directory:
            root=Path(directory); source_checkpoint(root/'source.pt')
            path=train_evaluator(root/'source.pt',root/'evaluator',device='cpu',epochs=1,batch_size=4,limit=4)
            train_evaluator(root/'source.pt',root/'evaluator',device='cpu',epochs=2,batch_size=4,limit=4,resume=True)
            evaluator=FrozenEvaluator.load(path,'cpu'); self.assertTrue(all(not p.requires_grad for p in evaluator.model.parameters()))
            logits,features=evaluator.evaluate(torch.zeros(2,1,28,28)); self.assertEqual((logits.shape,features.shape),((2,10),(2,64)))
    @patch('src.mnist_experiments.data.MNIST',side_effect=fake_mnist)
    def test_full_tiny_suite_resume(self,_):
        with TemporaryDirectory() as directory:
            root=Path(directory); source_checkpoint(root/'source.pt')
            save_pt(root/'evaluator.pt',dict(format='mnist_evaluator_v1',model=EvaluatorCNN().state_dict(),metadata={'train_indices':torch.arange(12),'validation_indices':torch.arange(12,16),'validation_accuracy':0.}))
            cfg=merge_dict(DEFAULT,{'checkpoint':str(root/'source.pt'),'device':'cpu','population_size':2,'ddim_steps':3,'batch_size':4,
             'evaluation':{'train_limit':8,'validation_limit':4,'knn_k':2,'include_test':True,'evaluator_checkpoint':str(root/'evaluator.pt')},
             'search':{'schedule':{'kind':'explicit','timesteps':[5,0]},'steps_per_level':1,'final_steps':1,'merge':{'radius_schedule':'constant','radius':0.}},
             'stability':{'enabled':True,'directions':2,'steps':1,'scales':[.01]},'curvature':{'enabled':True,'directions':2},'visualization':{'merge_comparison':True}})
            rows=run_suite(cfg,root/'suite'); self.assertEqual(len(rows),8)
            for row in rows:
                if row['method'].startswith('matched_'): self.assertLessEqual(row['nfe'],rows[0]['nfe']); self.assertLess(rows[0]['nfe']-row['nfe'],3)
                self.assertGreater(row['diagnostic_nfe'],0)
            a=torch.load(root/'suite/original/arrays.pt',weights_only=True); b=torch.load(root/'suite/ddim/arrays.pt',weights_only=True)
            torch.testing.assert_close(a['initial_noise'],b['initial_noise'])
            with patch('src.mnist_experiments.run.load_model',side_effect=AssertionError('Completed suite should skip model loading')):
                self.assertEqual(rows,run_suite(cfg,root/'suite',resume=True))
            self.assertTrue((root/'suite/original/neighbors_train_test.png').exists())
            original_stamp=(root/'suite/original/complete.json').stat().st_mtime_ns
            (root/'suite/complete.json').unlink(); (root/'suite/matched_cluster/complete.json').unlink()
            rows2=run_suite(cfg,root/'suite',resume=True)
            self.assertEqual(len(rows2),8); self.assertEqual(rows2[0]['nfe'],rows[0]['nfe'])
            self.assertEqual(original_stamp,(root/'suite/original/complete.json').stat().st_mtime_ns)
            cfg['search']['alpha']*=2
            with self.assertRaises(ValueError): run_suite(cfg,root/'suite',resume=True)
    def test_paired_statistics(self):
        rows=[dict(suite_id=str(seed),condition_id='a',variant='a',method=method,space='pixel',dimension=784,model_seed=42,search_seed=seed,initial_noise_seed=seed,pixel_knn_mean=value) for seed in (1,2) for method,value in [('original',1.),('ddim',3.)]]
        _,paired=summarize(rows,50); self.assertEqual(paired[0]['mean'],2.); self.assertEqual(paired[0]['ci95'],[2.,2.])
    def test_legacy_synthetic_levine(self):
        from src.models.diffusion import DiffusionModel
        from levine.src.models.diffusion import DiffusionModel as Levine
        for cls in (DiffusionModel,Levine): self.assertTrue(torch.isfinite(cls(dim=2,T=12,hidden_dims=[8],device=torch.device('cpu')).score(torch.zeros(2,2),0)).all())
    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_mps(self): self.backend('mps')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda(self): self.backend('cuda')
    def backend(self,device):
        counted=CountedModel(MNISTDiffusion(T=12,base_channels=8,time_dim=16,device=device),4); initial=counted.ddim(gaussian_noise(2,784,1),3)
        cfg=merge_dict(DEFAULT['search'],{'schedule':{'kind':'explicit','timesteps':[5,0]},'steps_per_level':1,'final_steps':1})
        self.assertTrue(torch.isfinite(run_search(counted,initial,cfg,PixelSpace())['candidates']).all())
if __name__=='__main__': unittest.main()
