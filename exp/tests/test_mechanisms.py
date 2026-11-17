"""Mechanism invariants independent of CUDA and fitted research outcomes."""
import unittest
import numpy as np
import pandas as pd
from exp.exp4.audit import clique_change, neighbor_sets, regimes
from exp.exp5.method_audit import group_indices, shares_and_regimes, reference_nodes


class MechanismTests(unittest.TestCase):
    def test_fixed_clique_intervention_matches_full_forward_input(self):
        rng=np.random.default_rng(4)
        own=rng.random((3,4,7))
        labels=np.array([0,0,0,0,1,1,1])
        targets=np.array([0,4])
        peers=np.array([[1,2],[5,6]])
        donors=np.array([[5,6],[1,2]])
        for replacement in (None,donors):
            actual=clique_change(own,labels,targets,peers,replacement)
            for row,j in enumerate(targets):
                modified=own.copy()
                modified[:,:,peers[row]]=0 if replacement is None else own[:,:,replacement[row]]
                group=labels==labels[j]
                expected=.5*(modified[:,:,group].mean(2)-own[:,:,group].mean(2))
                np.testing.assert_allclose(actual[:,:,row],expected,atol=1e-7)

    def test_matched_size_and_disconnected_donors(self):
        rng=np.random.default_rng(9)
        raw=rng.poisson(1,(50,15)).astype(float)
        labels=np.repeat(np.arange(3),5)
        targets=np.array([0,6,11])
        sets,donors,degree,_=neighbor_sets(raw,labels,targets,10,42)
        np.testing.assert_array_equal(degree,[4,4,4])
        for selected in sets.values():
            np.testing.assert_array_equal((selected>=0).sum(1),degree)
            for row,j in enumerate(targets):
                peers=selected[row,selected[row]>=0]
                self.assertNotIn(j,peers)
                self.assertTrue(np.all(labels[peers]==labels[j]))
        for row,j in enumerate(targets):
            self.assertTrue(np.all(labels[donors[row,donors[row]>=0]]!=labels[j]))

    def test_regimes_are_exclusive_and_evidence_does_not_use_outcomes(self):
        y=np.array([[[0.,2.,0.,1.]]]); p=np.ones_like(y)
        evidence=np.array([[0.,0.,2.,2.]])
        neighbors=np.array([[2.,2.,2.,0.]])
        ids,names,strata=regimes(y,p,np.zeros_like(p),evidence,neighbors,(1.,1.))
        self.assertEqual(names[ids[0,0,0]],'Spurious transfer')
        self.assertEqual(names[ids[0,0,1]],'Substitutive transfer')
        _,_,changed=regimes(y+10,p,np.zeros_like(p),evidence,neighbors,(1.,1.))
        np.testing.assert_array_equal(strata,changed)

    def test_feature_groups_partition_components(self):
        indexes=np.concatenate(group_indices(14))
        np.testing.assert_array_equal(np.sort(indexes),np.arange(66))
        absolute=np.ones((3,66)); shares,_=shares_and_regimes(absolute,14)
        np.testing.assert_allclose(shares.sum(1),1.)

    def test_legal_references_ignore_future_values(self):
        rng=np.random.default_rng(12)
        raw=rng.poisson(np.linspace(.1,9,100),(92,100)).astype(float)
        attrs=pd.DataFrame({'latitude':rng.random(100),'longitude':rng.random(100)})
        original=reference_nodes(raw,attrs,50)
        raw[50:]=1e9
        self.assertEqual(original,reference_nodes(raw,attrs,50))
        self.assertEqual(len(set(original)),2)


if __name__=='__main__':
    unittest.main()
