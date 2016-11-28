import sys
import operator
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.metrics import accuracy_score,f1_score
from sklearn.pipeline import _name_estimators
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class MetaClassifier(BaseEstimator, ClassifierMixin):
    """ A combined multi-class classifier

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
    Note: Classifiers need to be well caliberated

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
                weights=None, method='majority_voting'):
        self.classifiers = classifiers
        self.named_classifiers = {k:v for k,v in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
        self.method = method

    def fit(self, X_list, y_list, nested=True):
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
        if nested:
            X_list = map(np.vstack, X_list)
            y_list = map(np.hstack, y_list)
        self.lablenc_.fit(y_list[0]) # make sure both y vectors have both the classes
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for i,clf in enumerate(self.classifiers):
            fitted_clf = clone(clf).fit(X_list[i],
                                self.lablenc_.transform(y_list[i]))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X_list):
        """ Predict class labels.
        Parameters
        ----------
        X_list : List of {array-like, sparse matrix},
                length = number of classifiers
                List of matrices of training samples

        Returns
        -------
        maj_vote : array-like, shape = [n_samples]  
                   Predicted class labels
        """

        num_clfs = len(self.classifiers_)
        preds = []
        for index, X in enumerate(X_list):
            pred = [np.mean(self.classifiers_[index].predict_proba(P), axis=0) for P in X]
            preds.append(pred)
        preds = np.asarray(preds)
        weighted_proba = np.average(preds, axis=0, weights=self.weights) 
        maj_vote = np.argmax(weighted_proba, axis=1)
        return maj_vote


    def predict_proba(self, X_list, get_all=False):
        """ Predict class probabilities.
        Parameters
        ----------
        X_list : List of {array-like, sparse matrix},
                length = number of classifiers
                List of matrices of training samples

        Returns
        -------
        weighted_proba : array-like,shape = [n_samples, n_classes]           
                         Weighted average probability 
                         for each class per sample.
        """

        num_clfs = len(self.classifiers_)
        preds = []
        for index, X in enumerate(X_list):
            pred = [np.mean(self.classifiers_[index].predict_proba(P), axis=0) for P in X]
            preds.append(pred)
        preds = np.asarray(preds)
        weighted_proba = np.average(preds, axis=0, weights=self.weights)
        if get_all: 
            return preds[0], preds[1], weighted_proba
        return weighted_proba

    def score(self, Xs, y_true, scoring='f1'):
        """
        Returns the f1 score by default

        Parameters
        ----------
        Xs : List of {array-like, sparse matrix},
             length = number of classifiers
             List of matrices of training samples

        y_true: Single vectors of target class labels.
        
        """
        y_true = np.asarray(y_true)
        if scoring == 'f1':
            return f1_score(y_true,self.predict(Xs),average='binary')
        elif scoring == 'accuracy':
            return accuracy_score(y_true, self.predict(Xs))
        

class LateFusionClassifier(BaseEstimator, ClassifierMixin):
    """
    Plurality/Majority voting based Combined Classifier. Supports both
    single feature set/multiple feature set based Classification.
    """
    def __init__(self,classifiers,vote='classlabel',weights=None):
        self.classifiers = classifiers  # list of classifiers
        self.vote = vote    # 'probability' or 'classlabel'
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.weights = weights  # weights for each of the classifiers

    def fit(self,Xs,ys): 
        """
        Trains on the data.
        Xs = [[], [], []]
        ys = [[], [], []]

        Returns: self
        """
        if isinstance(Xs,list) and isinstance(ys,list):
            assert(len(Xs) == len(ys) == len(self.classifiers))
        self.classifiers_ = [] # store trained classifiers
        for idx, clf in enumerate(self.classifiers):
            fitted_clf = clone(clf).fit(Xs[idx],ys[idx])
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self,Xs):
        """
        Predicts new data instances.

        Args:
            Xs = [[], [], []]

        Returns:
            maj_vote: Predicted class
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X),axis=1)

        else: # classlabel
            predictions = np.asarray([clf.predict(Xs[mode_idx]) for mode_idx,clf in enumerate(self.classifiers_)]).T
            print '\n',predictions
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x,
                        weights=self.weights)),axis=1,arr=predictions)
        return maj_vote


    def predict_proba(self, Xs, get_all=False):
        """
        Predicts new data instances.

        Args:
            Xs = [[], [], []]

        Returns:
            avg_proba: Average probabilities of the class
        """

        probas = np.asarray([clf.predict_proba(Xs[mode_idx]) 
                            for mode_idx,clf in enumerate(self.classifiers_)])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        if get_all:
            return probas[0], probas[1], probas[2], avg_proba
        return avg_proba


    def score(self,Xs,y_true,scoring='f1'):
        """
        Returns the weighted F1-score (default)
        """
        if scoring == 'f1':
            return f1_score(y_true,self.predict(Xs),average='binary')
        elif scoring == 'accuracy':
            return accuracy_score(y_true, self.predict(Xs))


