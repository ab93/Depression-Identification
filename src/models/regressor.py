import sys
import operator
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.externals import six

class MetaRegressor(BaseEstimator, RegressorMixin):
    """ A combined multi-class regressor

    Parameters
    ----------
    regressors : array-like, shape = [n_classifiers]

    weights : array-like, shape = [n_classifiers]
    Optional, default: None
    If a list of `int` or `float` values are
    provided, the regressors are weighted by importance; 
    Uses uniform weights if `weights=None`.

    """

    def __init__(self, regressors, weights=None):
        self.regressors = regressors
        self.weights = weights

    def fit(self, X_list, y_list, nested=True):
        """ Fit regressors
        Parameters
        ----------
        X_list : List of {array-like, sparse matrix},
                length = number of classifiers
                List of matrices of training samples

        y_list : List of array-like,
                length = number of classifiers
                List of vectors of target class labels

        nested: Bool (default = True)

        Returns
        -------
        self : object
        """        
        assert(len(X_list) == len(y_list) == len(self.regressors))
        if (not isinstance(X_list,list)) or (not isinstance(y_list,list)):
            raise TypeError
            sys.exit()
        if nested:
            X_list = map(np.vstack, X_list)
            y_list = map(np.hstack, y_list)
        self.regressors_ = []
        for i,reg in enumerate(self.regressors):
            fitted_reg = clone(reg).fit(X_list[i], y_list[i])
            self.regressors_.append(fitted_reg)
        return self

    