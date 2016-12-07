import utils
import config
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from ..models import classifier

X_train, y_train, X_val, y_val = utils.get_single_mode_data(mode="acoustic",normalize='normalize')
y_true = map(int, map(np.mean, y_val[0]))
clf1 = DecisionTreeClassifier(class_weight={1:3},max_features = 13,min_samples_leaf=5, max_depth=5)
clf2 = LogisticRegression(C=1.0, class_weight={1: 3}, penalty='l1')
meta_clf = classifier.MetaClassifier(classifiers=[clf1,clf2], weights=[0.7,0.3])
meta_clf.fit(X_train, y_train)
f1_score = meta_clf.score(X_val, y_true)
print f1_score