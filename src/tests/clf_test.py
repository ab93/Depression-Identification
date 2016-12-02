import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from ..models.classifier import MetaClassifier, LateFusionClassifier
from ..feature_extract.read_labels import features
from ..main.classify import get_single_mode_data, get_multi_data

class MetaClassifierTest(unittest.TestCase):
    """
    Tests for the models.MetaClassifier class
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

    def _get_classifiers(self):
        clf1 = LogisticRegression(n_jobs=-1, class_weight={1:4})
        clf2 = LogisticRegression(n_jobs=-1, class_weight={1:4})
        return [clf1, clf2]

    def test_fit_predict(self):
        X_list, y_list = self._get_dummy_data()
        clfs = [LogisticRegression(C=100, penalty='l2'), LogisticRegression(C=10,penalty='l1')]
        meta_clf = MetaClassifier(clfs)
        meta_clf.fit(X_list,y_list)
        print "\npredict:",meta_clf.predict(X_list)

    def test_fit_predict_proba(self):
        X_list, y_list = self._get_dummy_data()
        clfs = [LogisticRegression(C=100, penalty='l2'), LogisticRegression(C=10,penalty='l1')]
        meta_clf = MetaClassifier(clfs)
        meta_clf.fit(X_list,y_list)
        print "\npredict:",meta_clf.predict_proba(X_list)

    def test_fit_score(self):
        X_list, y_list = self._get_dummy_data()
        clfs = [LogisticRegression(C=100, penalty='l2'), LogisticRegression(C=10,penalty='l1')]
        meta_clf = MetaClassifier(clfs)
        meta_clf.fit(X_list,y_list)
        y_true = np.array([1,0,0])
        print "\nscore:",meta_clf.score(X_list, y_true)

    def test_model(self):
        X_train, y_train, X_val, y_val = get_single_mode_data()
        y_true = map(int,map(np.mean,y_val[0]))
        
        clfs = self._get_classifiers()
        meta_clf = MetaClassifier(classifiers=clfs, weights=[0.9, 0.1])
        meta_clf.fit(X_train, y_train)
        
        print "\nTesting data..."
        preds = meta_clf.predict_proba(X_val, get_all=True) 
        
        print "F1-score: ", meta_clf.score(X_val, y_true)
        print "Accuracy: ", meta_clf.score(X_val, y_true, scoring='accuracy')

        for i in xrange(len(y_true)):
            print preds[0][i], preds[1][i], y_true[i]
    

class LateFusionClassifierTest(unittest.TestCase):
    """
    Tests for the models.LateFusionClassifierTest class
    """
    def _get_dummy_data(self):
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
        Xs, Ys = self._get_dummy_data()

        clf1 = self._get_fitted_clf(Xs[0],Ys[0])
        clf2 = self._get_fitted_clf(Xs[1],Ys[1])
        clf3 = self._get_fitted_clf(Xs[2],Ys[2])

        lf_clf = LateFusionClassifier(classifiers=[clf1,clf2,clf3])
        lf_clf.fit(Xs,Ys)
        print "\npredict:\n", lf_clf.predict(Xs)
        print "\npredict_proba:\n",lf_clf.predict_proba(Xs)

    def test_scores(self):
        Xs, Ys = self._get_dummy_data()

        clf1 = self._get_fitted_clf(Xs[0],Ys[0])
        clf2 = self._get_fitted_clf(Xs[1],Ys[1])
        clf3 = self._get_fitted_clf(Xs[2],Ys[2])

        lf_clf = LateFusionClassifier(classifiers=[clf1,clf2,clf3])
        lf_clf.fit(Xs,Ys)
        y_true = np.array([1,0,0])
        print "\npredict:\n", lf_clf.predict(Xs)
        print "\nscore:", lf_clf.score(Xs,y_true)

    def test_late_fusion_model(self):
        # Read the data
        Xs_train, ys_train, Xs_val, ys_val = get_multi_data()
        
        clf_A_D = LogisticRegression(C=1, penalty='l2', class_weight={1:4})
        clf_A_ND = LogisticRegression(C=0.001, penalty='l1', class_weight={1:4})

        clf_V_D = LogisticRegression(C=1.0, penalty='l2', class_weight={1:4})
        clf_V_ND = LogisticRegression(C=1.0, penalty='l2', class_weight={1:4})

        clf_L_D = LogisticRegression(C=1.0, penalty='l2', class_weight={1:3})
        clf_L_ND = LogisticRegression(C=1.0, penalty='l2', class_weight={1:3})

        clf_A = MetaClassifier(classifiers=[clf_A_D, clf_A_ND])
        clf_V = MetaClassifier(classifiers=[clf_V_D, clf_V_ND])
        clf_L = MetaClassifier(classifiers=[clf_L_D, clf_L_ND])
        
        lf_clf = LateFusionClassifier(classifiers=[clf_A, clf_V, clf_L], weights=[0.6,0.2,0.1])
        lf_clf.fit(Xs_train, ys_train)
        print lf_clf.predict(Xs_val)
        preds = lf_clf.predict_proba(Xs_val, get_all=True)
        y_true = map(int,map(np.mean,ys_val[0][0]))
        print lf_clf.score(Xs_val,y_true,scoring='f1')
        for i in xrange(len(y_true)):
            print preds[0][i], preds[1][i], preds[2][i], y_true[i]
    

if __name__ == '__main__':
    unittest.main()