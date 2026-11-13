"""CPU checks for portability, protocol correctness and published evidence."""
import ast
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'evaluation'))
from ablation import launcher
from ablation.protocol import GROUPS, config_for, alpha_at, stable_hash, validate_configs
from image_metrics import image_metrics
from geometry_metrics import surface_metrics, mesh_quality, transform_normals

spec=importlib.util.spec_from_file_location('public_eval',ROOT/'evaluation/evaluation.py')
evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
spec=importlib.util.spec_from_file_location('public_summary',ROOT/'scripts/summarize_results.py')
summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)


class ProtocolTests(unittest.TestCase):
    def test_released_numerical_source_identity(self):
        manifest=json.loads((ROOT/'results/chair/provenance.json').read_text())
        for rel,expected in manifest['model_source_files'].items():
            self.assertEqual(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),expected,rel)
        for name,expected in manifest['numerical_sources'].items():
            self.assertEqual(hashlib.sha256((ROOT/'ablation'/name).read_bytes()).hexdigest(),expected,name)

    def test_configs_and_schedule_boundaries(self):
        configs={g:config_for(g) for g in GROUPS}
        validate_configs(configs)
        for group,config in configs.items():
            self.assertEqual(json.loads((ROOT/'configs/chair'/(group+'.json')).read_text()),config)
            for epoch in (0,29,30,31,127,224,237,238):
                factor=min(1,max(0,(epoch-30)/194)) if config['schedule'] else 1
                self.assertEqual(alpha_at(config,epoch),(.005*factor if config['geometry'] else 0,.01*factor if config['texture'] else 0))
        self.assertEqual(238*3558,846804)

    def test_rgb_reference_and_range_protocol(self):
        from skimage.metrics import structural_similarity,peak_signal_noise_ratio
        rng=np.random.default_rng(817)
        y=rng.random((64,64,3));x=y+rng.normal(0,.08,y.shape)
        actual=image_metrics(x,y)
        self.assertAlmostEqual(actual['ssim'],structural_similarity(y,x.clip(0,1),data_range=1,channel_axis=-1,
            gaussian_weights=True,sigma=1.5,win_size=11,use_sample_covariance=False),places=12)
        self.assertAlmostEqual(actual['psnr_db'],peak_signal_noise_ratio(y,x.clip(0,1),data_range=1),places=12)
        self.assertGreater(actual['clip_fraction'],0)
        self.assertEqual(image_metrics(y,y)['psnr_db'],120)
        with self.assertRaises(ValueError):image_metrics(x[:63],y)

    def test_geometry_sanity(self):
        points=np.array([[0.,0,0],[1.,0,0],[0.,1,0]])
        normals=np.tile([0.,0,1.],(3,1))
        scores=surface_metrics(points,normals,points,normals,math.sqrt(2))
        self.assertEqual(scores['chamfer_l1_x10'],0)
        self.assertEqual(scores['fscore_01'],1)
        self.assertEqual(scores['normal_abs_cos'],1)
        self.assertTrue(np.allclose(transform_normals(normals,np.eye(3),[1,2,3]),normals))
        tri=np.array([[0.,0,0],[1.,0,0],[.5,math.sqrt(3)/2,0]])
        self.assertAlmostEqual(mesh_quality(tri,np.array([[0,1,2]]))['mean_triangle_quality'],1)

    def test_python39_and_stdlib_only_help(self):
        for path in ROOT.rglob('*.py'):
            if '.git' not in path.parts:
                ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,9))
        for script in ('train.py','evaluate.py','scripts/summarize_results.py'):
            result=subprocess.run([sys.executable,'-S',str(ROOT/script),'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_audited_result_means(self):
        import pandas as pd
        data=ROOT/'results/chair'
        rows=pd.read_csv(data/'all_9492_objects.csv')
        self.assertEqual(len(rows),9492)
        self.assertFalse(rows.duplicated(['group','object_id']).any())
        means=pd.read_csv(data/'seven_group_metrics_unrounded.csv').set_index('group')
        for key in ('icp_chamfer_l1_x10','icp_fscore_01','icp_fscore_02','icp_normal_abs_cos','image_psnr_db','image_ssim'):
            got=rows.groupby('group')[key].mean().reindex(means.index)
            self.assertTrue(np.allclose(got,means[key],rtol=1e-12,atol=1e-14),key)
        for key in ('seed','gt_subset_sha256','gt_diagonal'):
            self.assertTrue((rows.groupby('object_id')[key].nunique()==1).all())


class PreparationTests(unittest.TestCase):
    def test_prepare_freeze_and_evaluation_bundle(self):
        # Hash fixtures validate filesystem preparation; no real VGG, imagery,
        # weights or GPU are claimed by this test.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';workspace=root/'output';data.mkdir()
            names={'train':['train_'+str(i) for i in range(4744)],'val':['val_'+str(i) for i in range(64)],
                   'test':['test_'+str(i) for i in range(1356)]}
            for split,ids in names.items():(data/('softras_'+split+'.lst')).write_text('\n'.join(ids)+'\n')
            selected=[names['train'][i*4743//3] for i in range(4)]
            for obj in selected+names['val']:
                (data/obj/'image').mkdir(parents=True)
                for view in ((0,6,12,18,23) if obj in selected else (0,)):
                    (data/obj/'image'/('%04d.png'%view)).write_bytes(b'image-prepare-fixture')
            vgg=root/'vgg.pth';vgg.write_bytes(b'fixture-only')
            real_hash=launcher.sha256
            vgg_hash='397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'
            def fixture_hash(p):return vgg_hash if Path(p)==vgg else real_hash(p)
            before={p:real_hash(p) for p in data.rglob('*') if p.is_file()}
            with patch.object(launcher,'sha256',side_effect=fixture_hash):
                launcher.prepare(workspace,data,vgg)
                stage=workspace/'stages/chair_ablation_v1'
                launcher.verify(stage)
                with self.assertRaises(ValueError):launcher.prepare(workspace,data,vgg)
                frozen=stage/'ablation/engine.py';original=frozen.read_bytes()
                frozen.write_bytes(original+b'\n#changed\n')
                with self.assertRaises(RuntimeError):launcher.verify(stage)
                frozen.write_bytes(original)
            self.assertTrue(all(real_hash(p)==h for p,h in before.items()))
            self.assertIn('--array=0-6%2',(stage/'train_array.sbatch').read_text())
            for obj in names['test']:
                for rel in ('pointcloud.npz','cameras.npz','image/0000.png'):
                    p=data/obj/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'test-prepare-fixture')
            for group in GROUPS:
                p=stage/'runs'/group/'final.pkl';p.parent.mkdir(parents=True);p.write_bytes(b'checkpoint-prepare-fixture'*64)
            real_eval_hash=evaluation.sha256
            def eval_hash(p):return vgg_hash if Path(p)==vgg else real_eval_hash(p)
            with patch.object(evaluation,'sha256',side_effect=eval_hash):
                evaluation.prepare(workspace,None)
                eval_stage=workspace/'stages/chair_metrics_v1'
                prep,fingerprint=evaluation.verify(eval_stage)
            self.assertEqual(len(prep['objects']),1356)
            self.assertEqual(len(prep['pilot_objects']),8)
            bundle=eval_stage/'evaluate.pyz'
            with zipfile.ZipFile(bundle) as archive:
                ast.parse(archive.read('__main__.py').decode())
            result=subprocess.run([sys.executable,'-S',str(bundle),'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            with self.assertRaises(FileNotFoundError):evaluation.pilot_gate(eval_stage,fingerprint)

    def test_summary_recomputes_and_rejects_mismatched_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stage=root/'stage';stage.mkdir()
            ids=['o'+str(i) for i in range(1356)]
            prep={'objects':ids,'protocol':{'fixture':True},'checkpoints':{g:{'fixture':g} for g in GROUPS}}
            path=stage/'preparation.json';path.write_text(json.dumps(prep))
            fp=hashlib.sha256(path.read_bytes()).hexdigest()
            for group in GROUPS:
                rows=[{'group':group,'object_id':obj,'view':0,'seed':i,'gt_available':100000,'gt_used':10000,
                       'gt_subset_sha256':str(i),'pred_points':10000,'gt_diagonal':1.5,'icp_chamfer_l1_x10':.25} for i,obj in enumerate(ids)]
                report={'all_passed':True,'mode':'full','groups':[group],'evaluation_fingerprint':fp,'protocol':prep['protocol'],
                        'checkpoint_sources':{group:prep['checkpoints'][group]},'rows':rows,'new_objects_evaluated':1348,'objects_reused':8,
                        'metric_means':{group:{'icp_chamfer_l1_x10':.25}}}
                (stage/('final_'+group+'.json')).write_text(json.dumps(report))
            self.assertEqual(summary.summarize(stage,root/'summary')['rows'],9492)
            p=stage/'final_GTS.json';report=json.loads(p.read_text());report['rows'][0]['seed']=999
            p.write_text(json.dumps(report))
            with self.assertRaises(ValueError):summary.summarize(stage,root/'bad_summary')


if __name__=='__main__':unittest.main()
