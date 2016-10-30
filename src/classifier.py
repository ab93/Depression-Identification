from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

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

    def __init__(self, classifiers, vote='classlabel',
                weights='None', method='majority_voting'):
        self.classifiers = classifiers
        self.named_classifiers = {k:v for k,v in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
