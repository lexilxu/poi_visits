"""Checks for temporal leakage, future-data dependence, and tied-score AP."""
import unittest
import numpy as np
import pandas as pd
from exp.common.audit import window_arrays, canonical_partition
from exp.exp1.baselines import predictions, average_precision


class ProtocolTests(unittest.TestCase):
    def test_split_boundary_labels_are_disjoint(self):
        raw = np.ones((92,3))
        for t,h in [(14,7),(28,14),(28,28)]:
            origins,history,target,split,mask = window_arrays(raw,t,h)
            dates = {s:set(target[split==s][mask[split==s].any(2)].tolist())
                     for s in ('train','validation','test')}
            self.assertFalse(dates['train'] & dates['validation'])
            self.assertFalse(dates['validation'] & dates['test'])
            self.assertTrue(np.all(history[:,-1]<target[:,0]))
            self.assertFalse(mask[split=='validation',7:].any())

    def test_future_outcomes_cannot_change_earlier_forecasts(self):
        rng = np.random.default_rng(3)
        raw = rng.poisson(2, (92,4)).astype(float)
        dates = pd.date_range('2020-10-01', periods=92).strftime('%Y-%m-%d').to_numpy()
        before = predictions(raw,dates,14,7)
        changed = raw.copy()
        changed[75:] += 1000
        after = predictions(changed,dates,14,7)
        for name in before[0]:
            np.testing.assert_array_equal(before[0][name][before[1]<=75],
                                          after[0][name][after[1]<=75])
        self.assertEqual(before[-1],after[-1])

    def test_ties_in_precision_recall(self):
        labels = np.array([True,False,True,False])
        scores = np.ones(4)
        self.assertEqual(average_precision(labels,scores),.5)
        self.assertEqual(average_precision(labels,labels.astype(float)),1.)

    def test_partition_invariant_to_label_names(self):
        np.testing.assert_array_equal(canonical_partition(['a','b','a']),
                                      canonical_partition([41,2,41]))
        self.assertFalse(np.array_equal(canonical_partition(['a','b','a']),
                                        canonical_partition(['a','b','b'])))


if __name__=='__main__':
    unittest.main()
