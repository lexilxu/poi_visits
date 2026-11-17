"""Scientific invariants for the new controls, including leakage checks."""
import unittest
import numpy as np
from exp.common.controls import matched_neighbors, propagate_neighbors, affine_calibration, matched_far_threshold, spatial_coverage
from exp.exp1.extended_baselines import nlinear, tsb_states
from exp.common.audit import window_arrays
from exp.common.protocol import Protocol, feature_mask
from exp.common.evaluate import paired_intervals, aligned, export_events, load_events
from exp.exp1.queue_controls import plan, selected_fit
from pathlib import Path
import tempfile
import csv
import gzip


class ControlsTest(unittest.TestCase):
    def test_queue_preserves_all_seeds_and_uses_validation_only(self):
        manifest=plan('all')
        identifiers=[r['id'] for r in manifest['recipes']]
        self.assertEqual(len(identifiers),len(set(identifiers)))
        self.assertEqual(manifest['maximum_fits'],7*len(identifiers))
        self.assertEqual(manifest['seeds'],[42,43,44,45,46])
        candidates=[dict(lr=.001,validation_MAE=2.,test_MAE=0.),
                    dict(lr=.003,validation_MAE=1.,test_MAE=999.)]
        self.assertEqual(selected_fit(candidates)['lr'],.003)
        candidates[1]['test_MAE']=0.
        self.assertEqual(selected_fit(candidates)['lr'],.003)

    def test_density_cohort_ignores_future_and_breaks_ties_stably(self):
        from exp.common.inputs import density_indices
        raw=np.ones((92,20));changed=raw.copy();changed[50:]=9999
        for name in ('Dense25','Recent-dense25','Persist-dense25'):
            a,_=density_indices(raw,name);b,_=density_indices(changed,name)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(a,np.arange(15))

    def test_seed_average_is_loss_average_not_prediction_ensemble(self):
        from exp.common.paired import aggregate
        with tempfile.TemporaryDirectory() as folder:
            paths=[]
            for seed,value in zip(range(42,47),(0.,2.,0.,2.,0.)):
                path=Path(folder)/f'{seed}.npz';paths.append(path)
                np.savez(path,model='test',adjacency='Identity',seed=seed,
                         forecast_origin=np.array(['2020-12-04']),horizon_step=np.array([1]),
                         poi_id=np.array(['synthetic']),observed_visits=np.ones((1,1,1)),
                         predicted_visits=np.full((1,1,1),value),dataset_split=np.array(['test']))
            _,loss,_=aggregate(paths)
            self.assertEqual(float(loss.item()),1.)

    def test_paired_cluster_inference_and_event_schema(self):
        y=np.ones((22,7,4))
        p=y+2; q=y+1; valid=np.ones(y.shape,bool)
        valid[0,0,0]=False; y[0,0,0]=np.nan
        result=paired_intervals(y,p,q,valid,np.array([0,0,1,1]),repetitions=100)
        self.assertAlmostEqual(result['delta_MAE'],1.)
        for name in ('origin_7day','spatial_group','joint'):
            np.testing.assert_allclose(result[name+'_CI95'],[1.,1.])
        data=dict(model='test',adjacency='Identity',seed=42,
                  forecast_origin=np.arange(22).astype(str),horizon_step=np.arange(1,8),
                  poi_id=np.arange(4).astype(str),observed_visits=y,predicted_visits=p,
                  valid=valid,dataset_split=np.array(['test']*22))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'predictions.npz'; np.savez_compressed(path,**data)
            loaded=load_events(path); aligned(loaded,loaded)
            target=Path(folder)/'events.csv.gz'
            self.assertEqual(export_events(loaded,target,limit=2),2)
            with gzip.open(target,'rt',encoding='utf8') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]['observed_visits'],'')
            self.assertEqual(rows[1]['observed_visits'],'1.0')
            changed={**loaded,'poi_id':loaded['poi_id'][::-1]}
            with self.assertRaises(ValueError):
                aligned(loaded,changed)

    def test_complete_horizon_protocol(self):
        raw=np.ones((92,4))
        for t in (14,28):
            origins,history,target,split,valid=Protocol(T=t).windows(raw)
            self.assertEqual(int((split=='validation').sum()),8)
            self.assertEqual(int((split=='test').sum()),22)
            self.assertTrue((target[split=='train']<50).all())
            self.assertTrue((target[split=='validation']<64).all())
            self.assertTrue((target[split=='validation']>=50).all())
            self.assertTrue(np.all(history[:,-1]<target[:,0]))
        with self.assertRaises(ValueError):
            Protocol(T=28,H=28).windows(raw)

    def test_ablation_keeps_nonsemantic_features(self):
        only=feature_mask('only_naics2')
        np.testing.assert_array_equal(only[:4],np.ones(4))
        np.testing.assert_array_equal(only[12:20],np.ones(8))
        np.testing.assert_array_equal(only[20:28],np.zeros(8))
        np.testing.assert_array_equal(only[28:44],np.ones(16))
        self.assertEqual(feature_mask('without_parent').sum(),44)
        self.assertEqual(feature_mask('history_only').sum(),0)

    def test_equal_degree_and_disconnected_control(self):
        coarse = np.array([0,0,0,0,0,0,1,1,1])
        fine = np.array([0,0,0,1,1,2,3,3,3])
        graphs = matched_neighbors(coarse,fine,5,42)
        degrees = [(peers >= 0).sum(1) for peers in graphs.values()]
        for degree in degrees[1:]:
            np.testing.assert_array_equal(degree,degrees[0])
        x = np.arange(9.,dtype=float)[None,:]
        peers = graphs['NAICS-4']
        changed = x.copy(); changed[0,7] = 9999.
        self.assertEqual(propagate_neighbors(x,peers)[0,0],
                         propagate_neighbors(changed,peers)[0,0])
        np.testing.assert_array_equal(peers[:,0] == -1, fine == 2)

    def test_nlinear_selection_cannot_use_test_outcomes(self):
        rng = np.random.default_rng(32)
        raw = rng.poisson(2.,(92,30)).astype(float)
        _,history,target,split,valid = window_arrays(raw,14,7)
        p,info,w = nlinear(raw,history,target,split,valid)
        changed = raw.copy(); changed[64:] += 1000
        q,other,v = nlinear(changed,history,target,split,valid)
        np.testing.assert_allclose(w,v)
        np.testing.assert_allclose(p[split == 'validation'],q[split == 'validation'])
        self.assertEqual(info['ridge_penalty'],other['ridge_penalty'])

    def test_tsb_online_updates_and_missing_values(self):
        raw = np.array([[0.],[0.],[4.],[np.nan],[0.],[8.]])
        state = tsb_states(raw,.1,.2)
        self.assertEqual(state[2,0],0.)
        self.assertAlmostEqual(state[3,0],.8)
        self.assertEqual(state[4,0],state[3,0])
        self.assertAlmostEqual(state[5,0],.64)
        future = raw.copy(); future[-1] = 10000
        np.testing.assert_array_equal(state[:-1],tsb_states(future,.1,.2)[:-1])

    def test_calibration_and_tied_far(self):
        x = np.arange(1.,101.)/10
        result = affine_calibration(x,2*x+1)
        self.assertLess(result['validation_MAE'],1e-4)
        self.assertGreaterEqual(result['a'],0)
        scores = np.array([0.,0.,1.,1.,1.,2.])
        for far in (0.,.1,.2,.5):
            threshold = matched_far_threshold(scores,far)
            self.assertLessEqual((scores >= threshold).mean(),far)

    def test_activity_priority_respects_budget(self):
        y = np.array([[[1.,2.,5.,2.]]])
        p = np.array([[[9.,9.,0.,0.]]])
        actual,oracle,k,n = spatial_coverage(y,p,np.array([0,0,1,2]),1/3)
        self.assertEqual((k,n),(1,3))
        self.assertAlmostEqual(actual.item(),.3)
        self.assertAlmostEqual(oracle.item(),.5)


if __name__ == '__main__':
    unittest.main()
