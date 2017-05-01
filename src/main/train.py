import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from src.models.classifier import MetaClassifier, LateFusionClassifier
from src.models.regressor import MetaRegressor, LateFusionRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn import metrics
import config
from utils import get_multi_data, get_single_mode_data
import sys


class TrainRegressor(object):
    def __init__(self, category, feature_scale=True, modality='acoustic'):
        self.category = category
        self.feature_scale = feature_scale
        self.modality = modality

        self.reg_weights = [None, [0.7, 0.3], [0.3, 0.7]]

        # Ridge and Lasso params
        alpha_vals = np.logspace(-3, 3, num=5)
        lr_params = [{'alpha': alpha, 'normalize': False} for alpha in alpha_vals]

        # Decision Tree Regressor params
        max_features_ = np.arange(3, 20, 5)
        max_depths = np.arange(3, 6)
        min_samples_leaves = np.arange(2, 6)
        dt_params = [{'max_features': x, 'max_depth': y, 'min_samples_leaf': z, 'random_state': 42}
                     for x in max_features_ for y in max_depths
                     for z in min_samples_leaves]

        self.params = {'DT': dt_params,
                       'Ridge': lr_params,
                       'Lasso': lr_params}
        self.regs = {'DT': DecisionTreeClassifier,
                     'Ridge': Ridge,
                     'Lasso': Lasso}

    def grid_search_meta(self, count, model='DT'):
        x_train, y_train, x_val, y_val = get_single_mode_data(mode=self.modality, count=count,
                                                              category=self.category,
                                                              feature_scale=self.feature_scale,
                                                              problem_type='R')
        y_true_train = map(int, map(np.mean, y_train[0]))
        y_true_val = map(int, map(np.mean, y_val[0]))

        print "Performing regression grid search for {}".format(self.modality)

        with open(os.path.join(config.GRID_SEARCH_REG_DIR,
                               self.modality + '_' + model + '_' + self.category + '.txt'), 'w') as outfile:
            for reg_wt in self.reg_weights:
                for param1 in self.params[model]:
                    for param2 in self.params[model]:
                        reg_1 = self.regs[model](**param1)
                        reg_2 = self.regs[model](**param2)
                        meta_reg = MetaRegressor(regressors=[reg_1, reg_2], weights=reg_wt)
                        meta_reg.fit(x_train, y_train)
                        val_score = meta_reg.score(x_val, y_true_val)
                        train_score = meta_reg.score(x_train, y_true_train)
                        if not reg_wt:
                            reg_wt = [0.5, 0.5]
                        print "val RMSE:", val_score, "train RMSE:", train_score
                        outfile.write(str(reg_wt[0]) + ' ' + str(reg_wt[1]) + '\t' +
                                      str(param1) + '\t' + str(param2) + '\t' +
                                      str(val_score) + '\t' +
                                      str(train_score) + '\n')

    def grid_search_late_fusion(self, count):
        Xs_train, ys_train, Xs_val, ys_val = get_multi_data(count, self.category, feature_scale=self.feature_scale,
                                                            problem_type='R')
        y_true_val = map(int, map(np.mean, ys_val[0][0]))

        reg_A_1 = DecisionTreeRegressor(max_depth=5, max_features=3,
                                        min_samples_leaf=2, random_state=42)
        reg_A_2 = DecisionTreeRegressor(max_depth=5, max_features=13,
                                        min_samples_leaf=5, random_state=42)

        reg_V_1 = DecisionTreeRegressor(max_depth=5, max_features=8,
                                        min_samples_leaf=2, random_state=42)
        reg_V_2 = DecisionTreeRegressor(max_depth=5, max_features=18,
                                        min_samples_leaf=2, random_state=42)

        reg_L_1 = DecisionTreeRegressor(max_depth=5, max_features=3,
                                        min_samples_leaf=2, random_state=42)
        reg_L_2 = DecisionTreeRegressor(max_depth=4, max_features=3,
                                        min_samples_leaf=2, random_state=42)

        reg_A = MetaRegressor(regressors=[reg_A_1, reg_A_2])
        reg_V = MetaRegressor(regressors=[reg_V_1, reg_V_2], weights=[0.7, 0.3])
        reg_L = MetaRegressor(regressors=[reg_L_1, reg_L_2])

        mode_weights = [None, [0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.4, 0.4, 0.2],
                        [0.5, 0.4, 0.1], [0.4, 0.5, 0.1], [0.25, 0.25, 0.5],
                        [0.2, 0.7, 0.1], [0.2, 0.55, 0.25]]

        with open(os.path.join(config.GRID_SEARCH_REG_DIR, '_' + self.category + '_lf.txt'), 'w') as outfile:
            outfile.write('A_wt' + ',' + 'V_wt' + ',' + 'L_wt' + ',' + 'score' + '\n')
            for mode_wt in mode_weights:
                lf_reg = LateFusionRegressor(regressors=[reg_A, reg_V, reg_L], weights=mode_wt)
                lf_reg.fit(Xs_train, ys_train)
                score = lf_reg.score(Xs_val, y_true_val)
                print mode_wt
                print "LF:", lf_reg.predict(Xs_val)
                print "A :", lf_reg.regressors_[0].predict(Xs_val[0])
                print "V :", lf_reg.regressors_[1].predict(Xs_val[1])
                print "L :", lf_reg.regressors_[2].predict(Xs_val[2])
                print "Y: ", np.array(y_true_val)
                if not mode_wt:
                    mode_wt = [0.3, 0.3, 0.3]
                outfile.write(str(mode_wt[0]) + ',' + str(mode_wt[1]) + ',' +
                              str(mode_wt[2]) + ',' + str(score) + '\n')
                print "LF:", score
                print "A :", lf_reg.regressors_[0].score(Xs_val[0], y_true_val)
                print "V :", lf_reg.regressors_[1].score(Xs_val[1], y_true_val)
                print "L :", lf_reg.regressors_[2].score(Xs_val[2], y_true_val), '\n'


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

    def grid_search_meta(self, count, select, model='DT'):
        x_train, y_train, x_val, y_val = get_single_mode_data(mode=self.modality, count=count, select=select,
                                                              category=self.category,
                                                              feature_scale=self.feature_scale)

        y_true_train = map(int, map(np.mean, y_train[0]))
        y_true_val = map(int, map(np.mean, y_val[0]))

        print "Performing classification grid search for {}".format(self.modality)

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

    def grid_search_late_fusion(self,count,select):
        Xs_train, ys_train, Xs_val, ys_val = get_multi_data(count, select, self.category, feature_scale=self.feature_scale)
        y_true_val = map(int, map(np.mean, ys_val[0][0]))
        y_true_train = map(int, map(np.mean, ys_train[0][0]))

        clf_A_1 = DecisionTreeClassifier(class_weight={1: 4}, max_depth=5, max_features=3,
                                         min_samples_leaf=2, random_state=42)
        clf_A_2 = DecisionTreeClassifier(class_weight={1: 4}, max_depth=5, max_features=13,
                                         min_samples_leaf=5, random_state=42)

        clf_V_1 = DecisionTreeClassifier(class_weight={1: 3}, max_depth=5, max_features=8,
                                         min_samples_leaf=2, random_state=42)
        clf_V_2 = DecisionTreeClassifier(class_weight={1: 3}, max_depth=5, max_features=18,
                                         min_samples_leaf=2, random_state=42)

        clf_L_1 = DecisionTreeClassifier(class_weight={1: 4}, max_depth=5, max_features=3,
                                         min_samples_leaf=2, random_state=42)
        clf_L_2 = DecisionTreeClassifier(class_weight={1: 4}, max_depth=4, max_features=3,
                                         min_samples_leaf=2, random_state=42)

        clf_A = MetaClassifier(classifiers=[clf_A_1, clf_A_2])
        clf_V = MetaClassifier(classifiers=[clf_V_1, clf_V_2], weights=[0.7, 0.3])
        clf_L = MetaClassifier(classifiers=[clf_L_1, clf_L_2])

        mode_weights = [None, [0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.4, 0.4, 0.2],
                        [0.5, 0.4, 0.1], [0.4, 0.5, 0.1], [0.25, 0.25, 0.5],
                        [0.2, 0.7, 0.1], [0.2, 0.55, 0.25]]

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

    def plot_learning_curve(self):
        steps = np.arange(56, 187, 10)
        # steps = [186]
        if self.modality == 'acoustic':
            clf = MetaClassifier(classifiers=[DecisionTreeClassifier(class_weight={1: 4}, max_depth=5, max_features=3,
                                                                     min_samples_leaf=2, random_state=42),
                                              DecisionTreeClassifier(class_weight={1: 4}, max_depth=5, max_features=13,
                                                                     min_samples_leaf=5, random_state=42)])
        elif self.modality == 'visual':
            clf = MetaClassifier(classifiers=[DecisionTreeClassifier(class_weight={1: 3}, max_depth=5, max_features=8,
                                                                     min_samples_leaf=2, random_state=42),
                                              DecisionTreeClassifier(class_weight={1: 3}, max_depth=5, max_features=18,
                                                                     min_samples_leaf=2, random_state=42)],
                                 weights=[0.7, 0.3])
        elif self.modality == 'linguistic':
            clf = MetaClassifier(classifiers=[DecisionTreeClassifier(class_weight={1: 4}, max_depth=5, max_features=3,
                                                                     min_samples_leaf=2, random_state=42),
                                              DecisionTreeClassifier(class_weight={1: 4}, max_depth=4, max_features=3,
                                                                     min_samples_leaf=2, random_state=42)])

        train_scores, val_scores = [], []
        for train_count in steps:
            x_train, y_train, x_val, y_val = get_single_mode_data(mode=self.modality, count=train_count,
                                                                  category=self.category,
                                                                  feature_scale=self.feature_scale)
            y_true_train = map(int, map(np.mean, y_train[0]))
            y_true_val = map(int, map(np.mean, y_val[0]))

            clf.fit(x_train, y_train)

            val_scores.append(clf.score(x_val, y_true_val))
            train_scores.append(clf.score(x_train, y_true_train))

        print val_scores
        print train_count

        plt.figure()
        plt.plot(steps, train_scores, 'o-', color="r",
                 label="Training score")
        plt.plot(steps, val_scores, 'o-', color="g",
                 label="Validation score")

        plt.title('Learning Curve for {}'.format(self.modality))
        plt.xlabel("Training examples")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.legend(loc='best')
        plt.show()

    def plot_roc(self):
        Xs_train, ys_train, Xs_val, ys_val = get_multi_data(count, self.category, feature_scale=self.feature_scale)
        y_true_val = map(int, map(np.mean, ys_val[0][0]))

        clf_A_1 = DecisionTreeClassifier(class_weight={1: 4}, max_depth=5, max_features=3,
                                         min_samples_leaf=2, random_state=42)
        clf_A_2 = DecisionTreeClassifier(class_weight={1: 4}, max_depth=5, max_features=13,
                                         min_samples_leaf=5, random_state=42)

        clf_V_1 = DecisionTreeClassifier(class_weight={1: 3}, max_depth=5, max_features=8,
                                         min_samples_leaf=2, random_state=42)
        clf_V_2 = DecisionTreeClassifier(class_weight={1: 3}, max_depth=5, max_features=18,
                                         min_samples_leaf=2, random_state=42)

        clf_L_1 = DecisionTreeClassifier(class_weight={1: 4}, max_depth=5, max_features=3,
                                         min_samples_leaf=2, random_state=42)
        clf_L_2 = DecisionTreeClassifier(class_weight={1: 4}, max_depth=4, max_features=3,
                                         min_samples_leaf=2, random_state=42)

        probs = {}

        clf_A = MetaClassifier(classifiers=[clf_A_1, clf_A_2])
        clf_A.fit(Xs_train[0], ys_train[0])
        probs['acoustic'] = clf_A.predict_proba(Xs_val[0])[:, 1]

        clf_V = MetaClassifier(classifiers=[clf_V_1, clf_V_2], weights=[0.7, 0.3])
        clf_V.fit(Xs_train[1], ys_train[1])
        probs['visual'] = clf_V.predict_proba(Xs_val[1])[:, 1]

        clf_L = MetaClassifier(classifiers=[clf_L_1, clf_L_2])
        clf_L.fit(Xs_train[2], ys_train[2])
        probs['linguistic'] = clf_L.predict_proba(Xs_val[2])[:, 1]

        lf_clf = LateFusionClassifier(classifiers=[clf_A, clf_V, clf_L])
        lf_clf.fit(Xs_train, ys_train)
        probs['lateFusion'] = lf_clf.predict_proba(Xs_val)[:, 1]

        labels = ['acoustic', 'visual', 'linguistic', 'lateFusion']
        fpr, tpr, roc_auc = {}, {}, {}
        plt.figure()
        colors = ['aqua', 'darkorange', 'cornflowerblue', 'green']
        for lbl, c in zip(labels, colors):
            fpr[lbl], tpr[lbl], _ = metrics.roc_curve(y_true_val, probs[lbl], pos_label=1)
            roc_auc[lbl] = metrics.auc(fpr[lbl], tpr[lbl])
            plt.plot(fpr[lbl], tpr[lbl], color=c, lw=2,
                     label='ROC curve of {0} (area = {1:0.2f})'.format(lbl, roc_auc[lbl]))

        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


if __name__ == '__main__':

    # import argparse
    #
    # parser = argparse.ArgumentParser(description='Run Grid Search for Classification/Regression. ')
    # parser.add_argument('-type', default='C', options=)

    if len(sys.argv) == 2:
        count = sys.argv[1]
    else:
        count = "all"

    # trn = TrainClassifier(category='PN', feature_scale=False, modality='acoustic')
    # trn.grid_search_meta(count, model='DT')
    # trn.grid_search_late_fusion(count)
    # trn.plot_roc()
    # trn.plot_learning_curve()

    trn = TrainRegressor(category='PN', feature_scale=False, modality='acoustic')
    # trn.grid_search_meta(count, model='Ridge')
    trn.grid_search_late_fusion(count)

    # print "Selecting features...\n"
    # feature_select.feature_select("C")

    # print "Normalizing features...\n"
    # normalize_features(select=select)
    # norm = 'normalize'

    # print "Performing Grid Search for visual...\n"
    # grid_search_meta(mode='visual', category='PN', normalize=norm)

    # print "Performing Grid Search for acoustic...\n"
    # grid_search_meta(mode='acoustic', category='PN', normalize=norm)
    # print "Performing Grid Search for linguistic...\n"
    # grid_search_meta(mode='linguistic', category='PN', normalize=norm)
    # print "Performing Grid Search for Late Fusion...\n"
    # grid_search_lf(category='PN', normalize=norm)
    count = "all" # Input the number of training data to use or 'all'
    select = "select" # Input 'select' or 'all'
    if len(sys.argv) >= 2:
        arg1 = sys.argv[1].split('=')
        key = arg1[0]
        if key == 'count':
            count = arg1[1]
        else:
            select = arg1[1]
        if len(sys.argv) >= 3:
            arg2 = sys.argv[2].split('=')
            key = arg2[0]
            if key == 'count':
                count = arg2[1]
            else:
                select = arg2[1]
            if len(sys.argv)>3:
                print "Usage: python train.py [count={count|all}] [select={select|all}]"

    print count
    print select
    raw_input()
    trn = TrainClassifier(category='PN', feature_scale=False, modality='linguistic')
    trn.grid_search_meta(count, select, model='DT')
    trn.grid_search_late_fusion(count,select)

