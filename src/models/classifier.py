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


    def predict_proba(self, X_list):
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
        return weighted_proba

    def score(self, X_list, y_list, scoring='f1'):
        """
        Returns the f1 score by default
        """
        pass #TODO
        

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
            assert(len(X_list) == len(y_list) == len(self.classifiers))
        self.classifiers_ = []
        for i in range(len(Xs)):
            classifiers = []
            for clf in self.classifiers:
                fitted_clf = clone(clf).fit(Xs[i],ys[i])
                classifiers.append(fitted_clf)
            self.classifiers_.append(classifiers)
        return self

    def predict(self,X):
        """
        Predicts new data instances.

        Args:
            X: Matrix of feature vectors and instances
                OR
               List consisting of matrices of feature vectors and instances
               (if multi_data = True)

        Returns:
            maj_vote: Predicted class
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X),axis=1)

        else: # classlabel
            # If multi_data is set True
            if self.multi_data:
                for clf_grp_index in range(len(self.classifiers_)):
                    x = X[clf_grp_index]
                    classifiers = self.classifiers_[clf_grp_index]
                    pred = np.asarray([clf.predict(x) for clf in classifiers]).T
                    if clf_grp_index == 0:
                        predictions = pred.copy()
                    else:
                        predictions = np.hstack((predictions,pred))
                    maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x,
                                weights=self.weights)),axis=1,arr=predictions)

            # If multi_data is set False
            else:
                predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
                maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x,
                            weights=self.weights)),axis=1,arr=predictions)

        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote


    def predict_proba(self,X):
        """
        Returns probability estimates for test data
        """
        # If multi_data is set True
        if self.multi_data:
            for clf_grp_index in range(len(self.classifiers_)):
                x = X[clf_grp_index]
                classifiers = self.classifiers_[clf_grp_index]
                p = np.asarray([clf.predict_proba(x) for clf in classifiers])
                if clf_grp_index == 0:
                    probas = p.copy()
                else:
                    probas = np.vstack((probas,p))
        # If multi_data is set False
        else:
            probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])

        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba


    def get_params(self,deep=True):
        """
        Returns the parameters of the base classifiers.
        """
        if not deep:
            return super(PluralityVoteClassifier,self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key,value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out


    def score(self,X,y,sample_weight=None,scoring='f1'):
        """
        Returns the weighted F1-score (default)
        """
        if scoring == 'f1':
            return f1_score(y,self.predict(X),average='weighted',sample_weight=sample_weight,pos_label=None)
        elif scoring == 'accuracy':
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


# Just for debugging and testing

def main():
    # for discriminative
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

    clfs = [LogisticRegression(C=100, penalty='l2'), LogisticRegression(C=10,penalty='l1')]
    meta_clf = MetaClassifier(clfs)
    meta_clf.fit(X,y)
    print "predict:",meta_clf.predict(X)
    print "predict_proba:",meta_clf.predict_proba(X)

    X_acou, y_acou = [x1,x2], [y1,y2]
    X_vis, y_vis = [x2,x1], [y2,y1]
    X_lin, y_lin = [x1,x1], [y1,y1]



if __name__ == '__main__':
    main()

