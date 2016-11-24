import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from ..models.classifier import MetaClassifier, LateFusionClassifier

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
        print "\npredict:",meta_clf.predict(X_list)

    def test_fit_predict_proba(self):
        X_list, y_list = self._set_test_data()
        clfs = [LogisticRegression(C=100, penalty='l2'), LogisticRegression(C=10,penalty='l1')]
        meta_clf = MetaClassifier(clfs)
        meta_clf.fit(X_list,y_list)
        print "\npredict:",meta_clf.predict_proba(X_list)

    def test_fit_score(self):
        X_list, y_list = self._set_test_data()
        clfs = [LogisticRegression(C=100, penalty='l2'), LogisticRegression(C=10,penalty='l1')]
        meta_clf = MetaClassifier(clfs)
        meta_clf.fit(X_list,y_list)
        y_true = np.array([1,0,0])
        print "\nscore:",meta_clf.score(X_list, y_true)


class LateFusionClassifierTest(unittest.TestCase):
    """
    Tests for the models.LateFusionClassifierTest class
    """
    def _set_test_data(self):
        x1 = np.array([ np.array([[1,5,7], [1,2,4], [1,8,9]]),
                np.array([[2,8,6], [2,0,3]]), 
                np.array([[3,7,5], [3,4,3], [3,9,7]]) 
                ])
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

        X_acou, y_acou = [x1,x2], [y1,y2]
        X_vis, y_vis = [x1,x2], [y1,y2]
        X_lin, y_lin = [x1,x2], [y1,y2]
        
        return [X_acou, X_vis, X_lin], [y_acou, y_vis, y_lin]

    def _get_fitted_clf(self,Xs,ys):
        clfs = [LogisticRegression(C=100, penalty='l2'), LogisticRegression(C=10,penalty='l1')]
        meta_clf = MetaClassifier(clfs)
        meta_clf.fit(Xs,ys)
        return meta_clf

    def test_fit_predict(self):
        Xs, Ys = self._set_test_data()

        clf1 = self._get_fitted_clf(Xs[0],Ys[0])
        clf2 = self._get_fitted_clf(Xs[1],Ys[1])
        clf3 = self._get_fitted_clf(Xs[2],Ys[2])

        lf_clf = LateFusionClassifier(classifiers=[clf1,clf2,clf3])
        lf_clf.fit(Xs,Ys)
        print "\npredict:\n", lf_clf.predict(Xs)
        print "\npredict_proba:\n",lf_clf.predict_proba(Xs)

    def test_scores(self):
        Xs, Ys = self._set_test_data()

        clf1 = self._get_fitted_clf(Xs[0],Ys[0])
        clf2 = self._get_fitted_clf(Xs[1],Ys[1])
        clf3 = self._get_fitted_clf(Xs[2],Ys[2])

        lf_clf = LateFusionClassifier(classifiers=[clf1,clf2,clf3])
        lf_clf.fit(Xs,Ys)
        y_true = np.array([1,0,0])
        print "\npredict:\n", lf_clf.predict(Xs)
        print "\nscore:", lf_clf.score(Xs,y_true)
    

if __name__ == '__main__':
    unittest.main()