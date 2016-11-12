import sys
import operator
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.pipeline import _name_estimators
from sklearn.svm import SVC

class MetaClassifier(BaseEstimator, ClassifierMixin):
    """ A combined multi-class classifier classifier

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]

    vote : str, {'classlabel', 'probability'}
    Default: 'classlabel'

    weights : array-like, shape = [n_classifiers]
    Optional, default: None
    If a list of `int` or `float` values are
    provided, the classifiers are weighted by
    importance; Uses uniform weights if `weights=None`.

    method: str, {'stacking', 'majority_voting'}
    Default: 'majority_voting'

    """

    def __init__(self, classifiers, vote='probability',
                weights='None', method='majority_voting'):
        self.classifiers = classifiers
        self.named_classifiers = {k:v for k,v in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
        self.method = method

    def fit(self, X_list, y_list):
        """ Fit classifiers.
        Parameters
        ----------
        X_list : List of {array-like, sparse matrix},
                length = number of classifiers
                List of matrices of training samples

        y_list : List of array-like,
                length = number of classifiers
                List of vectors of target class labels.

        Returns
        -------
        self : object
        """

        assert(len(X_list) == len(y_list) == len(self.classifiers))
        if (not isinstance(X_list,list)) or (not isinstance(y_list,list)):
            raise TypeError
            sys.exit()
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y_list[0]) # make sure both y vectors have both the classes
        self.classes_ = self.lablenc_.classes_
        X_list = map(np.vstack, X_list)
        y_list = map(np.stack, y_list)
        self.classifiers_ = []
        for i,clf in enumerate(self.classifiers):
            fitted_clf = clone(clf).fit(X_list[i],
                                self.lablenc_.transform(y_list[i]))
            self.classifiers_.append(fitted_clf)
        print self
        return self

    def predict(self, X_list):
        num_clfs = len(self.classifiers_)
        print len(self.classifiers_)
        #print self.classifiers_[1].predict_proba(X_list[1])
        #raw_input()
        if self.vote == 'probability':
            probas = np.asarray([clf.predict_proba(X_list[i])
                                for i,clf in enumerate(self.classifiers_)])
            


x1 = np.array([[1,2], [4,1], [3,1]])
x2 = np.array([[6,4],[8,9]])
y1 = np.array([0,1,1])
y2 = np.array([1,0])

X = [x1,x2]
y = [y1,y2]

x1 = np.array([ np.array([[1,5,7], [1,2,4], [1,8,9]]), 
                np.array([[2,8,6], [2,0,3]]), 
                np.array([[3,7,5], [3,4,3], [3,9,7]]) ])

x2 = np.array([np.array([[1,5,7], [1,2,4]]), 
            np.array([[2,8,6], [2,0,3], [2,5,5]]), 
            np.array([[3,7,5], [3,4,3], [3,9,7]])])


clfs = [SVC(probability=True), SVC(probability=True)]
meta_clf = MetaClassifier(clfs)
meta_clf.fit(X,y)
#meta_clf.predict(X)
