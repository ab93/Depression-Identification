import sys
import operator
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.externals import six
from sklearn.metrics import r2_score, mean_absolute_error

class MetaRegressor(BaseEstimator, RegressorMixin):
    """ A combined multi-class regressor

    Parameters
    ----------
    regressors : array-like, shape = [n_regressors]

    weights : array-like, shape = [n_regressors]
    Optional, default: None
    If a list of `int` or `float` values are
    provided, the regressors are weighted by importance;
    Uses uniform weights if `weights=None`.

    """

    def __init__(self, regressors, weights=None):
        self.regressors = regressors
        self.weights = weights

    def fit(self, Xs, ys, nested=True):
        """ Fit regressors
        Parameters
        ----------
        Xs : List of {array-like, sparse matrix},
                length = number of regressors
                List of matrices of training samples

        ys : List of array-like,
                length = number of regressors
                List of vectors of target class labels

        nested: Bool (default = True)

        Returns
        -------
        self : object
        """
        assert(len(Xs) == len(ys) == len(self.regressors))
        if (not isinstance(Xs,list)) or (not isinstance(ys,list)):
            raise TypeError
            sys.exit()
        if nested:
            Xs = map(np.vstack, Xs)
            ys = map(np.hstack, ys)
        self.regressors_ = []
        for i,reg in enumerate(self.regressors):
            fitted_reg = clone(reg).fit(Xs[i], ys[i])
            self.regressors_.append(fitted_reg)
        return self

    def predict(self, Xs):
        """ Predict class labels.
        Parameters
        ----------
        Xs : List of {array-like, sparse matrix},
                length = number of regressors
                List of matrices of training samples

        Returns
        -------
        weighted_pred : array-like, shape = [n_samples]
                Predicted (weighted) target values
        """

        num_regs = len(self.regressors_)
        preds = []
        for index, X in enumerate(Xs):
            pred = [np.mean(self.regressors_[index].predict(P), axis=0) for P in X]
            preds.append(pred)
        preds = np.asarray(preds)
        weighted_pred = np.average(preds, axis=0, weights=self.weights)
        return weighted_pred

    def score(self, Xs, y_true, scoring='mean_abs_error'):
        """
        Returns the R2 (Coefficient of Determination) score by default

        Parameters
        ----------
        Xs : List of {array-like, sparse matrix},
             length = number of regressors
             List of matrices of training samples

        y_true: Single vectors of true y values

        """
        y_true = np.asarray(y_true)
        if scoring == 'r2':
            return r2_score(y_true,self.predict(Xs))
        elif scoring == 'mean_abs_error':
            return mean_absolute_error(y_true, self.predict(Xs))


class LateFusionRegressor(BaseEstimator, RegressorMixin):
    """
    Weighted Combined Regressor
    """
    def __init__(self,regressors,weights=None):
        self.regressors = regressors  # list of regressors
        self.weights = weights  # weights for each of the regressors

    def fit(self,Xs,ys):
        """
        Trains on the data.
        Xs = [[], [], []] (one matrix for each mode)
        ys = [[], [], []]

        Returns: self
        """
        if isinstance(Xs,list) and isinstance(ys,list):
            assert(len(Xs) == len(ys) == len(self.regressors))
        self.regressors_ = [] # store trained regressors
        for idx, reg in enumerate(self.regressors):
            fitted_reg = clone(reg).fit(Xs[idx],ys[idx])
            self.regressors_.append(fitted_reg)
        return self

    def predict(self,Xs):
        """
        Predicts new data instances

        Args:
            Xs = [[], [], []]

        Returns:
            weighted_pred: Weighted prediction of the target
        """
        preds = []
        for mode_idx, reg in enumerate(self.regressors_):
            preds.append(reg.predict(Xs[mode_idx]))
        preds = np.asarray(preds)
        weighted_preds = np.average(preds, axis=0, weights=self.weights)
        return weighted_preds

    def score(self, Xs, y_true, scoring='mean_abs_error'):
        """
        Returns the R2 (Coefficient of Determination) score by default

        Parameters
        ----------
        Xs : List of {array-like, sparse matrix},
             length = number of regressors
             List of matrices of training samples

        y_true: Single vectors of true y values

        """
        y_true = np.asarray(y_true)
        if scoring == 'r2':
            return r2_score(y_true,self.predict(Xs))
        elif scoring == 'mean_abs_error':
            return mean_absolute_error(y_true, self.predict(Xs))
