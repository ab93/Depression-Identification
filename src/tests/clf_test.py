import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from ..models.classifier import MetaClassifier

class MetaClassifierTest(unittest.TestCase):
    """
    Tests for the models.MetaClassifier class
    """
    def _set_test_data(self):
        x1 = np.array([ np.array([[1,5,7], [1,2,4], [1,8,9]]), # [r1,r2,r3] for p1
                np.array([[2,8,6], [2,0,3]]),  # [r1,r2] for p2
                np.array([[3,7,5], [3,4,3], [3,9,7]]) # [r1,r2,r3] for p3
                ])

        # for non discriminative
        x2 = np.array([ np.array([[1,5,7], [1,2,4]]), 
                        np.array([[2,8,6], [2,0,3], [2,5,5]]), 
                        np.array([[3,7,5], [3,4,3], [3,9,7]])
                        ])

        y1 = np.array([ np.array([1,1,1]),
                        np.array([1,1]),
                        np.array([0,0,0])
                        ])

        y2 = np.array([ np.array([0,0]), 
                        np.array([0,0,0]), 
                        np.array([1,1,1])
                        ])

        X = [x1,x2]
        y = [y1,y2]
        return X,y

    def test_fit_predict(self):
        X_list, y_list = self._set_test_data()
        clfs = [LogisticRegression(C=100, penalty='l2'), LogisticRegression(C=10,penalty='l1')]
        meta_clf = MetaClassifier(clfs)
        meta_clf.fit(X_list,y_list)
        print "predict:",meta_clf.predict(X_list)

    def test_fit_predict_proba(self):
        X_list, y_list = self._set_test_data()
        clfs = [LogisticRegression(C=100, penalty='l2'), LogisticRegression(C=10,penalty='l1')]
        meta_clf = MetaClassifier(clfs)
        meta_clf.fit(X_list,y_list)
        print "predict:",meta_clf.predict_proba(X_list)


if __name__ == '__main__':
    unittest.main()