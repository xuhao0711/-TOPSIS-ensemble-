import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from topsis_ensemble import topsis, assert_isolation, select_on_training

class Tests(unittest.TestCase):
    def test_direction_negative_r2(self):
        t = pd.DataFrame({'model':['good','bad'], 'MAE':[1,2], 'RMSE':[2,3], 'R2':[-1,-4]})
        r = topsis(t)
        self.assertEqual(r.iloc[0].model, 'good')
        self.assertAlmostEqual(r.iloc[0].closeness, 1)
        self.assertAlmostEqual(r.ensemble_weight.sum(), 1)
    def test_ties_zero_and_invalid(self):
        t = pd.DataFrame({'model':['a','b'], 'MAE':[0,0], 'RMSE':[0,0], 'R2':[0,0]})
        np.testing.assert_allclose(topsis(t).closeness, .5)
        with self.assertRaises(ValueError): topsis(t, [-1,1,1])
        t.loc[0, 'MAE'] = np.nan
        with self.assertRaises(ValueError): topsis(t)
    def test_parent_and_duplicate_leakage(self):
        a = pd.DataFrame({'row_id':[1], 'x':[2]})
        b = pd.DataFrame({'row_id':[3], 'parent_id_1':[1], 'x':[4]})
        with self.assertRaises(ValueError): assert_isolation(a,b,['x'])
        b = pd.DataFrame({'row_id':[3], 'x':[2]})
        with self.assertRaises(ValueError): assert_isolation(a,b,['x'])
    def test_train_only_and_augmentation_scope(self):
        X = np.arange(24.)[:,None]; y = X[:,0] + np.sin(X[:,0]); groups = np.repeat(np.arange(8),3)
        seen = []
        def aug(x,y):
            seen.append(len(x)); return x,y
        spaces = {'ridge': {'pipeline': Ridge(), 'param_grid': {'alpha':[.1,1]}}}
        s = select_on_training(X,y,groups,spaces,3,2,augment=aug)
        self.assertTrue(any(n < len(X) for n in seen))
        self.assertTrue(np.isfinite(s['oof']['ridge']).all())
        r = s['ranking'].copy(deep=True)
        s['ensemble'].predict(np.array([[100.]]))
        pd.testing.assert_frame_equal(r,s['ranking'])

if __name__ == '__main__': unittest.main()
