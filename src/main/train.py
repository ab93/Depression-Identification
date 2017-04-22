import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.models.classifier import MetaClassifier, LateFusionClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import config
from utils import get_multi_data, get_single_mode_data
from ..helpers.normalized_features import normalize_features


class TrainClassifier(object):
    def __init__(self, category, feature_scale=True, modality='acoustic'):
        self.category = category
        self.feature_scale = feature_scale
        self.modality = modality

        self.class_weights = np.arange(3, 6)
        self.clf_weights = [None, [0.7, 0.3], [0.3, 0.7]]

        # Logistic Regression params
        c_values = np.logspace(-3, 3, num=5)
        penalties = ('l1', 'l2')
        max_ent_params = [{'C': x, 'penalty': y} for x in c_values for y in penalties]

        # Decision Tree params
        max_features_ = np.arange(3, 20, 5)
        max_depths = np.arange(3, 6)
        min_samples_leaves = np.arange(2, 6)
        dt_params = [{'max_features': x, 'max_depth': y, 'min_samples_leaf': z, 'random_state': 42}
                     for x in max_features_ for y in max_depths
                     for z in min_samples_leaves]# Decision Tree params
        max_features_ = np.arange(3, 20, 5)
        max_depths = np.arange(3, 6)
        min_samples_leaves = np.arange(2, 6)
        dt_params = [{'max_features': x, 'max_depth': y, 'min_samples_leaf': z, 'random_state':42}
                     for x in max_features_ for y in max_depths
                     for z in min_samples_leaves]

        # AdaBoost params
        base_estimators = [DecisionTreeClassifier(max_depth=1),
                           DecisionTreeClassifier(max_depth=3)]
        num_estimators = np.arange(50, 200, 50)
        boost_params = [{'base_estimator': x, 'n_estimators': y}
                        for x in base_estimators for y in num_estimators]

        self.params = {'DT': dt_params,
                       'LR': max_ent_params,
                       'AdaBoost': boost_params}
        self.clfs = {'DT': DecisionTreeClassifier,
                     'LR': LogisticRegression,
                     'AdaBoost': AdaBoostClassifier}

    def grid_search_meta(self, model='DT'):
        x_train, y_train, x_val, y_val = get_single_mode_data(mode=self.modality,
                                                              category=self.category,
                                                              feature_scale=self.feature_scale)
        y_true_train = map(int, map(np.mean, y_train[0]))
        y_true_val = map(int, map(np.mean, y_val[0]))

        with open(os.path.join(config.GRID_SEARCH_CLF_DIR,
                               self.modality + '_' + model + '_' + self.category + '.txt'), 'w') as outfile:
            for clf_wt in self.clf_weights:
                for class_wt in self.class_weights:
                    for param1 in self.params[model]:
                        for param2 in self.params[model]:
                            clf_1 = self.clfs[model](class_weight={1: class_wt}, **param1)
                            clf_2 = self.clfs[model](class_weight={1: class_wt}, **param2)
                            meta_clf = MetaClassifier(classifiers=[clf_1, clf_2], weights=clf_wt)
                            meta_clf.fit(x_train, y_train)
                            val_f1_score = meta_clf.score(x_val, y_true_val)
                            train_f1_score = meta_clf.score(x_train, y_true_train)
                            if not clf_wt:
                                clf_wt = [0.5, 0.5]
                            print "val f1:", val_f1_score, "train f1:", train_f1_score
                            outfile.write(str(clf_wt[0]) + ' ' + str(clf_wt[1]) + '\t' + str(class_wt) + '\t' +
                                          str(param1) + '\t' + str(param2) + '\t' +
                                          str(val_f1_score) + '\t' +
                                          str(train_f1_score) + '\n')

    def grid_search_late_fusion(self):
        Xs_train, ys_train, Xs_val, ys_val = get_multi_data(self.category, feature_scale=self.feature_scale)
        y_true_val = map(int, map(np.mean, ys_val[0][0]))
        y_true_train = map(int, map(np.mean, ys_train[0][0]))

        clf_A_1 = DecisionTreeClassifier(class_weight={1: 4}, max_depth=5, max_features=18,
                                         min_samples_leaf=2, random_state=42)
        clf_A_2 = DecisionTreeClassifier(class_weight={1: 4}, max_depth=5, max_features=3,
                                         min_samples_leaf=4, random_state=42)

        clf_V_1 = DecisionTreeClassifier(class_weight={1: 4}, max_depth=4, max_features=8,
                                         min_samples_leaf=2, random_state=42)
        clf_V_2 = DecisionTreeClassifier(class_weight={1: 4}, max_depth=5, max_features=3,
                                         min_samples_leaf=5, random_state=42)

        clf_L_1 = DecisionTreeClassifier(class_weight={1: 5}, max_depth=5, max_features=8,
                                         min_samples_leaf=2, random_state=42)
        clf_L_2 = DecisionTreeClassifier(class_weight={1: 5}, max_depth=5, max_features=13,
                                         min_samples_leaf=3, random_state=42)

        clf_A = MetaClassifier(classifiers=[clf_A_1, clf_A_2], weights=[0.3, 0.7])
        clf_V = MetaClassifier(classifiers=[clf_V_1, clf_V_2])
        clf_L = MetaClassifier(classifiers=[clf_L_1, clf_L_2])

        mode_weights = [None, [0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.4, 0.4, 0.2],
                        [0.5, 0.4, 0.1], [0.4, 0.5, 0.1], [0.25, 0.25, 0.5],
                        [0.2, 0.7, 0.1]]

        with open(os.path.join(config.GRID_SEARCH_CLF_DIR, '_' + self.category + '_lf.txt'), 'w') as outfile:
            outfile.write('A_wt' + ',' + 'V_wt' + ',' + 'L_wt' + ',' + 'f1_score' + '\n')
            for mode_wt in mode_weights:
                lf_clf = LateFusionClassifier(classifiers=[clf_A, clf_V, clf_L], weights=mode_wt)
                lf_clf.fit(Xs_train, ys_train)
                f1_score = lf_clf.score(Xs_val, y_true_val, scoring='f1')
                print mode_wt
                print "LF:", lf_clf.predict(Xs_val)
                print "A :", lf_clf.classifiers_[0].predict(Xs_val[0])
                print "V :", lf_clf.classifiers_[1].predict(Xs_val[1])
                print "L :", lf_clf.classifiers_[2].predict(Xs_val[2])
                print "Y: ", np.array(y_true_val)
                if not mode_wt:
                    mode_wt = [0.3, 0.3, 0.3]
                outfile.write(str(mode_wt[0]) + ',' + str(mode_wt[1]) + ',' +
                              str(mode_wt[2]) + ',' + str(f1_score) + '\n')
                print "LF:", f1_score
                print "A :", lf_clf.classifiers_[0].score(Xs_val[0], y_true_val)
                print "V :", lf_clf.classifiers_[1].score(Xs_val[1], y_true_val)
                print "L :", lf_clf.classifiers_[2].score(Xs_val[2], y_true_val), '\n'

                # print "A :", lf_clf.classifiers_[0].score(Xs_train[0], y_true_train)
                # print "V :", lf_clf.classifiers_[1].score(Xs_train[1], y_true_train)
                # print "L :", lf_clf.classifiers_[2].score(Xs_train[2], y_true_train), '\n'


if __name__ == '__main__':
    # print "Selecting features...\n"
    # feature_select.feature_select("C")

    # print "Normalizing features...\n"
    # normalize_features()
    # norm = 'normalize'

    # print "Performing Grid Search for visual...\n"
    # grid_search_meta(mode='visual', category='PN', normalize=norm)

    # print "Performing Grid Search for acoustic...\n"
    # grid_search_meta(mode='acoustic', category='PN', normalize=norm)
    # print "Performing Grid Search for linguistic...\n"
    # grid_search_meta(mode='linguistic', category='PN', normalize=norm)
    # print "Performing Grid Search for Late Fusion...\n"
    # grid_search_lf(category='PN', normalize=norm)

    trn = TrainClassifier(category='PN', feature_scale=False, modality='linguistic')
    # trn.grid_search_meta(model='DT')
    trn.grid_search_late_fusion()
