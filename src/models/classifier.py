import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression


class MetaClassifier(BaseEstimator, ClassifierMixin):
    """ A combined multi-class classifier

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
    vote : str, {'classlabel', 'probability'}
    weights : array-like, shape = [n_classifiers]
    
    If a list of `int` or `float` values are
    provided, the classifiers are weighted by
    importance; Uses uniform weights if `weights=None`.

    method: str, {'stacking', 'majority_voting'}

    """

    def __init__(self, classifiers, weights=None):
        self.classifiers = classifiers

        self.weights = weights
        self.lablenc_ = LabelEncoder()
        self.classes_ = None
        self.classifiers_ = []

    def fit(self, X_list, y_list):
        """ 
        Fit classifiers.
        Parameters
        ----------
        X_list : List of {array-like, sparse matrix},
                length = number of classifiers
                List of matrices of training samples

        y_list : List of array-like,
                length = number of classifiers
                List of vectors of target class labels

        Returns
        -------
        self : object
        """

        assert(len(X_list) == len(y_list) == len(self.classifiers))
        if (not isinstance(X_list, list)) or (not isinstance(y_list, list)):
            raise TypeError("Input is not of type list")

        X_list = map(np.vstack, [map(np.vstack, x) for x in X_list])
        y_list = map(np.hstack, [map(np.hstack, y) for y in y_list])

        # make sure both y vectors have both the classes
        self.lablenc_.fit(y_list[0])
        self.classes_ = self.lablenc_.classes_

        for i, clf in enumerate(self.classifiers):
            fitted_clf = clone(clf).fit(X_list[i], self.lablenc_.transform(y_list[i]))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X_list):
        """ Predict class labels
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
        num_examples = len(X_list[0])
        confidence_matrix = np.zeros((num_examples, 2))
        if self.weights is None:
            self.weights = [0.5, 0.5]

        for cat_idx, X in enumerate(X_list):
            for p_idx, p in enumerate(X):
                num_resp = len(p)
                confidence = np.zeros(2)  # for the two classes
                for r in p:
                    num_segments = r.shape[0]
                    num_depressed = np.count_nonzero(self.classifiers_[cat_idx].predict(r))
                    confidence[0] += (num_segments - num_depressed) / float(num_segments)  # non-depressed
                    confidence[1] += num_depressed / float(num_segments)  # depressed
                confidence_matrix[p_idx] += (self.weights[cat_idx] * confidence) / num_resp

        predictions = np.argmax(confidence_matrix, axis=1)
        return predictions

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
        num_examples = len(X_list[0])
        confidence_matrix = np.zeros((num_examples, 2))
        if self.weights is None:
            self.weights = [0.5, 0.5]

        for cat_idx, X in enumerate(X_list):
            for p_idx, p in enumerate(X):
                num_resp = len(p)
                confidence = np.zeros(2)  # for the two classes
                for r in p:
                    num_segments = r.shape[0]
                    num_depressed = np.count_nonzero(self.classifiers_[cat_idx].predict(r))
                    confidence[0] += (num_segments - num_depressed) / float(num_segments)  # non-depressed
                    confidence[1] += num_depressed / float(num_segments)  # depressed
                confidence_matrix[p_idx] += (self.weights[cat_idx] * confidence) / num_resp

        return confidence_matrix

    def score(self, Xs, ys_true, scoring='f1'):
        """
        Returns the f1 score by default

        Parameters
        ----------
        Xs : List of {array-like, sparse matrix},
             length = number of classifiers
             List of matrices of training samples

        ys_true: Single vectors of target class labels
        scoring: Type of metric used

        """
        if len(ys_true) != 2:
            raise ValueError("Length of ys_true is not 2")
        y_true = np.asarray(map(lambda x: int(x[0]), [map(np.mean, y) for y in ys_true[0]]))
        if scoring == 'f1':
            return f1_score(y_true, self.predict(Xs), average='binary')
        elif scoring == 'accuracy':
            return accuracy_score(y_true, self.predict(Xs))


class LateFusionClassifier(BaseEstimator, ClassifierMixin):
    """
    Plurality/Majority voting based Combined Classifier. Supports both
    single feature set/multiple feature set based Classification.
    """
    def __init__(self, classifiers, vote='soft', weights=None):
        self.classifiers = classifiers  # list of classifiers
        self.vote = vote    # soft or hard voting
        self.weights = weights  # weights for each of the classifiers
        self.classifiers_ = []  # store trained classifiers

    def fit(self, Xs, ys):
        """
        Trains on the data.
        Xs = [[], [], []]
        ys = [[], [], []]

        Returns: self
        """
        if isinstance(Xs, list) and isinstance(ys, list):
            assert(len(Xs) == len(ys) == len(self.classifiers))

        for idx, clf in enumerate(self.classifiers):
            fitted_clf = clone(clf).fit(Xs[idx], ys[idx])
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, Xs):
        """
        Predicts new data instances.

        Args:
            Xs = [[], [], []]

        Returns:
            maj_vote: Predicted class
        """

        # Hard voting
        if self.vote == 'hard':
            predictions = np.asarray([clf.predict(Xs[mode_idx]) for mode_idx, clf in
                                      enumerate(self.classifiers_)]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                                           axis=1,
                                           arr=predictions)
        # Soft voting
        else:
            maj_vote = np.argmax(self.predict_proba(Xs), axis=1)
        return maj_vote

    def predict_proba(self, Xs):
        """
        Predicts the probabilities/confidence of new data instances.

        Args:
            Xs = [[], [], []]

        Returns:
            avg_confidence: Average probabilities of the class
        """
        confidence_matrix = np.asarray([clf.predict_proba(Xs[mode_idx])
                                        for mode_idx, clf in enumerate(self.classifiers_)])
        avg_confidence = np.average(confidence_matrix, axis=0, weights=self.weights)
        return avg_confidence

    def score(self, Xs, ys_true, scoring='f1'):
        """
        Returns the weighted F1-score (default)
        """
        if len(ys_true) != 3:
            raise ValueError("Length of ys_true is not 3")
        if len(ys_true[0]) != 2:
            raise ValueError("Length of element inside ys_true is not 2")
        y_true = np.asarray(map(lambda x: int(x[0]), [map(np.mean, y) for y in ys_true[0][0]]))
        if scoring == 'f1':
            return f1_score(y_true, self.predict(Xs), average='binary')
        elif scoring == 'accuracy':
            return accuracy_score(y_true, self.predict(Xs))


if __name__ == '__main__':
    x1 = [[np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9]])],
          [np.array([[1, 5, 1], [0, 9, 2]])]]

    x2 = [[np.array([[1, 2, 7], [4, 2, 6]]), np.array([[7, 0, 9]])],
          [np.array([[2, 2, 3], [4, 5, 6]])]]

    y1 = [[np.array([1, 1]), np.array([1])],
          [np.array([0, 0])]]

    y2 = [[np.array([1, 0]), np.array([0])],
          [np.array([0, 1])]]

    data, labels = [x1, x2], [y1, y2]

    meta_clf = MetaClassifier(classifiers=[LogisticRegression(), LogisticRegression()])
    meta_clf.fit(data, labels)
    print "Testing MetaClassifier"
    print meta_clf.predict(data)
    print meta_clf.predict_proba(data)
    print meta_clf.score(data, labels)

    meta_clf1 = MetaClassifier(classifiers=[LogisticRegression(), LogisticRegression()])
    meta_clf2 = MetaClassifier(classifiers=[LogisticRegression(), LogisticRegression()])
    meta_clf3 = MetaClassifier(classifiers=[LogisticRegression(), LogisticRegression()])
    lf_clf = LateFusionClassifier(classifiers=[meta_clf1, meta_clf2, meta_clf3], vote='hard')
    data, labels = [[x1, x2], [x2, x1], [x1, x1]], [[y1, y2], [y2, y1], [y1, y1]]
    lf_clf.fit(data, labels)
    print "Testing LateFusionClassifier"
    print lf_clf.predict(data)
    print lf_clf.predict_proba(data)
    print lf_clf.score(data, labels)

