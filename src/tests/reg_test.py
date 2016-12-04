import unittest
import numpy as np
from sklearn.linear_model import LinearRegression
from ..models.regressor import MetaRegressor
from ..feature_extract.read_labels import features
from ..main.classify import get_single_mode_data, get_multi_data

class MetaRegressorTest(unittest.TestCase):
    """
    Tests for models.MetaRegressor class
    """
    def _get_dummy_data(self):
        x1 = np.array([ np.array([[1,5,7], [1,2,4], [1,8,9]]), # [r1,r2,r3] for p1
                np.array([[2,8,6], [2,0,3]]),  # [r1,r2] for p2
                np.array([[3,7,5], [3,4,3], [3,9,7]]) # [r1,r2,r3] for p3
                ])

        # for non discriminative
        x2 = np.array([ np.array([[1,5,7], [1,2,4]]), 
                        np.array([[2,8,6], [2,0,3], [2,5,5]]), 
                        np.array([[3,7,5], [3,4,3], [3,9,7]])
                        ])

        y1 = np.array([ np.array([5.53,5.53,5.53]),
                        np.array([7.2,7.2]),
                        np.array([2,74,2.74])
                        ])

        y2 = np.array([ np.array([6.3,6.3]), 
                        np.array([3.9,3.9,3.9]), 
                        np.array([9.2,9.2,9.2])
                        ])
        X = [x1,x2]
        y = [y1,y2]
        return X,y

    def _get_regressors(self):
        reg1 = LinearRegression(normalize=True)
        reg2 = LinearRegression(normalize=True)
        return [reg1, reg2]

    def test_fit_predict(self):
        X_list, y_list = self._get_dummy_data()
        y_true = np.array([5.0,11.0,13.5])
        meta_reg = MetaRegressor(self._get_regressors())
        meta_reg.fit(X_list,y_list)
        preds = meta_reg.predict(X_list)
        print "preds:\n", preds
        print "R2 score:\n",meta_reg.score(X_list,y_true)
        print "Mean abs error:\n",meta_reg.score(X_list,y_true,scoring='mean_abs_error')




if __name__ == '__main__':
    unittest.main()