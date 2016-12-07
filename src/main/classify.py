import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.models.classifier import MetaClassifier, LateFusionClassifier
from src.feature_extract import read_labels
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import _name_estimators
from ..feature_extract.read_labels import features
import config
import feature_select
from utils import get_multi_data, get_single_mode_data
from ..helpers.normalized_features import normalize_features
import grid_search_dt_lr
from sklearn.externals import joblib
from copy import deepcopy
def grid_search_meta(mode='acoustic',category='PN',normalize='normalize'):
    X_train, y_train, X_val, y_val = get_single_mode_data(mode=mode,
                                    category=category, normalize=normalize)

    X_data = deepcopy(X_train)
    X_data[0].extend(X_val[0])
    X_data[1].extend(X_val[1])
    y_data = deepcopy(y_train)
    y_data[0].extend(y_val[0])
    y_data[1].extend(y_val[1])

    # Set y_true for validation
    y_true_val = map(int,map(np.mean,y_val[0]))

    # Set parameters for GridSearch
    class_weights = np.arange(3,6)
    clf_weights = [None, [0.7,0.3], [0.3,0.7]]

    C_vals = np.logspace(-3,3,num=5)
    gammas = np.logspace(-3,3,num=5)
    penalties = ('l1','l2')


    max_features_ = np.arange(3,20,5)
    max_depths = np.arange(3,6)
    min_samples_leaves = np.arange(2,6)

    estimators = [LogisticRegression, DecisionTreeClassifier]
    named_clfs = _name_estimators(estimators)
    named_clfs = [x[0] for x in named_clfs]
    dt_vals = [{'max_features': x, 'max_depth': y, 'min_samples_leaf': z}
                for x in max_features_ for y in max_depths for z in min_samples_leaves]
    svm_vals = [{'C':x, 'gamma':y, 'probability':True} for x in C_vals for y in gammas]
    max_ent_vals = [{'C':x, 'penalty':y } for x in C_vals for y in penalties]
    #all_params = [max_ent_vals, svm_vals]

    res_LR,res_DT = grid_search_dt_lr.ret_func()
    clf_weights_refined = res_LR[0]
    clf_weights_refined = [x.split(" ") for x in clf_weights_refined]
    #print clf_weights_refined
    #raw_input()
    class_weights_refined = res_LR[1]
    dt_vals_refined = res_DT[1]
    lr_vals_refined = [str({'C':x, 'penalty':y }) for x in res_LR[2] for y in res_LR[4]]
    all_params_refined = [lr_vals_refined,dt_vals_refined]
    # print all_params_refined
    # raw_input()
    max_f1 = 0.0
    max_clf1 = None
    max_clf2 = None
    max_weights = []
    #with open(os.path.join(config.GRID_SEARCH_CLF_DIR, mode + '_test_' + category + '.txt'),'w') as outfile:
    #    outwriter=csv.writer(outfile, delimiter='\t')

    for clf_wt in clf_weights_refined:
        temp = []
        for i in clf_wt:
            temp.append(float(i))
        for class_wt in class_weights_refined:
            for i,clf1 in enumerate(estimators):
                for j,clf2 in enumerate(estimators):
                    for param1 in all_params_refined[i]:
                        for param2 in all_params_refined[j]:
                            clf_1 = clf1(class_weight={1:class_wt}, **eval(param1))
                            clf_2 = clf2(class_weight={1:class_wt}, **eval(param2))
                            meta_clf = MetaClassifier(classifiers=[clf_1, clf_2], weights=temp)
                            meta_clf.fit(X_train, y_train)
                            f1_score = meta_clf.score(X_val, y_true_val)
                            if f1_score > max_f1:
                                max_f1 = f1_score
                                max_clf1 = clf_1
                                max_clf2 = clf_2
                                max_weights = temp[:]

                            print i,j,f1_score,temp

    meta_clf = MetaClassifier(classifiers=[max_clf1, max_clf2], weights=max_weights)
    meta_clf.fit(X_data, y_data)
    joblib.dump(meta_clf, os.path.join(config.GRID_SEARCH_CLF_DIR, mode + '_pickle' + category + '.pkl'))


    # with open(os.path.join(config.GRID_SEARCH_CLF_DIR, mode + '_SVM_' + category + '.txt'),'w') as outfile:
    #     for clf_wt in clf_weights:
    #         for class_wt in class_weights:
    #             for param1 in svm_vals:
    #                 for param2 in svm_vals:
    #                     clf_1 = SVC(class_weight={1:class_wt}, **param1)
    #                     clf_2 = SVC(class_weight={1:class_wt}, **param2)
    #                     meta_clf = MetaClassifier(classifiers=[clf_1, clf_2], weights=clf_wt)
    #                     meta_clf.fit(X_train, y_train)
    #                     f1_score = meta_clf.score(X_val, y_true_val)
    #                     print f1_score
    #                     if not clf_wt:
    #                         clf_wt = [0.5, 0.5]
    #                     outfile.write(str(clf_wt[0]) + ' ' + str(clf_wt[1]) + '\t' +
    #                     str(param1) + '\t' + str(param2) + '\t' +
    #                     str(class_wt) + '\t' + str(f1_score) +'\n')

    # with open(os.path.join(config.GRID_SEARCH_CLF_DIR, mode + '_DT_' + category + '.txt'),'w') as outfile:
    #     for clf_wt in clf_weights:
    #         for class_wt in class_weights:
    #             for param1 in dt_vals:
    #                 for param2 in dt_vals:
    #                     clf_1 = DecisionTreeClassifier(class_weight={1:class_wt}, **param1)
    #                     clf_2 = DecisionTreeClassifier(class_weight={1:class_wt}, **param2)
    #                     meta_clf = MetaClassifier(classifiers=[clf_1, clf_2], weights=clf_wt)
    #                     meta_clf.fit(X_train, y_train)
    #                     f1_score = meta_clf.score(X_val, y_true_val)
    #                     print f1_score
    #                     if not clf_wt:
    #                         clf_wt = [0.5, 0.5]
    #                     outfile.write(str(clf_wt[0]) + ' ' + str(clf_wt[1]) + '\t' +
    #                     str(param1) + '\t' + str(param2) + '\t' +
    #                     str(class_wt) + '\t' + str(f1_score) +'\n')

    # with open(os.path.join(config.GRID_SEARCH_CLF_DIR, mode + '_' + category + '.csv'),'w') as outfile:
    #     for p1 in penalties:
    #         for p2 in penalties:
    #             for clf_wt in clf_weights:
    #                 for class_wt in class_weights:
    #                     for C1 in C_vals:
    #                         for C2 in C_vals:
    #                             clf_D = LogisticRegression(C=C1, penalty=p1, n_jobs=-1, class_weight={1:class_wt})
    #                             clf_ND = LogisticRegression(C=C2, penalty=p2, n_jobs=-1, class_weight={1:class_wt})
    #                             meta_clf = MetaClassifier(classifiers=[clf_D, clf_ND], weights=clf_wt)
    #                             meta_clf.fit(X_train, y_train)
    #                             f1_score = meta_clf.score(X_val, y_true_val)
    #                             if not clf_wt:
    #                                 clf_wt = [0.5, 0.5]
    #                             outfile.write(str(clf_wt[0]) + ' ' + str(clf_wt[1]) + ',' +
    #                                         str(class_wt) + ',' + str(C1) + ',' + str(C2) + ','
    #                                         + p1 + ',' + p2 + ','
    #                                         + str(f1_score) + '\n')
    #                             print f1_score




def grid_search_late_fusion(category='PN', normalize='normalize'):
    Xs_train, ys_train, Xs_val, ys_val = get_multi_data(category, normalize=normalize)
    y_true_val = map(int, map(np.mean, ys_val[0][0]))
    X_data = deepcopy(Xs_train)
    X_data[0][0].extend(Xs_val[0][0])
    X_data[0][1].extend(Xs_val[0][1])
    X_data[1][0].extend(Xs_val[1][0])
    X_data[1][1].extend(Xs_val[1][1])
    X_data[2][0].extend(Xs_val[2][0])
    X_data[2][1].extend(Xs_val[2][1])

    y_data = deepcopy(ys_train)
    y_data[0][0].extend(ys_val[0][0])
    y_data[0][1].extend(ys_val[0][1])
    y_data[1][0].extend(ys_val[1][0])
    y_data[1][1].extend(ys_val[1][1])
    y_data[2][0].extend(ys_val[2][0])
    y_data[2][1].extend(ys_val[2][1])

    clf_a =  joblib.load(os.path.join(config.GRID_SEARCH_CLF_DIR  + '/acoustic_pickle' + category + '.pkl'))
    clf_v =  joblib.load(os.path.join(config.GRID_SEARCH_CLF_DIR  + '/visual_pickle' + category + '.pkl'))
    clf_l = joblib.load(os.path.join(config.GRID_SEARCH_CLF_DIR + '/linguistic_pickle' + category + '.pkl'))
    max_f1 = 0.0
    max_clf = None
    mode_weights = [None, [0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.4, 0.4, 0.2],
                    [0.5, 0.4, 0.1], [0.4, 0.5, 0.1], [0.25, 0.25, 0.5]]
    for mode_wt in mode_weights:
        lf_clf = LateFusionClassifier(classifiers=[clf_a, clf_v, clf_l], weights=mode_wt)
        lf_clf.fit(Xs_train, ys_train)
        f1_score = lf_clf.score(Xs_val, y_true_val, scoring='f1')
        if not mode_wt:
            mode_wt = [0.3, 0.3, 0.3]
        print f1_score
        if(f1_score > max_f1):
            max_f1 = f1_score
            max_clf = lf_clf

    max_clf.fit(X_data, y_data)
    joblib.dump(max_clf, os.path.join(config.GRID_SEARCH_CLF_DIR + '/late_fusion_picklePN.pkl'))





def grid_search_lf(category='PN', normalize='normalize'):
    Xs_train, ys_train, Xs_val, ys_val = get_multi_data(category, normalize=normalize)
    y_true_val = map(int,map(np.mean,ys_val[0][0]))

    # For Positive Negative
    if category == 'PN':
        clf_A_1 = LogisticRegression(C=1, penalty='l2', class_weight={1:4})
        clf_A_2 = LogisticRegression(C=0.001, penalty='l1', class_weight={1:4})

        clf_V_1 = LogisticRegression(C=31.623, penalty='l1', class_weight={1:4})
        clf_V_2 = LogisticRegression(C=1.0, penalty='l2', class_weight={1:4})

        clf_L_1 = LogisticRegression(C=0.031623, penalty='l1', class_weight={1:4})
        clf_L_2 = LogisticRegression(C=0.001, penalty='l1', class_weight={1:4})

    # For Disc Non-disc
    else:
        clf_A_1 = LogisticRegression(C=1, penalty='l1', class_weight={1:4})
        clf_A_2 = LogisticRegression(C=0.0316, penalty='l1', class_weight={1:4})

        clf_V_1 = LogisticRegression(C=0.001, penalty='l2', class_weight={1:4})
        clf_V_2 = LogisticRegression(C=1000, penalty='l2', class_weight={1:4})

        clf_L_1 = LogisticRegression(C=0.031623, penalty='l1', class_weight={1:4})
        clf_L_2 = LogisticRegression(C=0.001, penalty='l1', class_weight={1:4})

    clf_A = MetaClassifier(classifiers=[clf_A_1, clf_A_2], weights=[0.3, 0.7])
    clf_V = MetaClassifier(classifiers=[clf_V_1, clf_V_2])
    clf_L = MetaClassifier(classifiers=[clf_L_1, clf_L_2])

    mode_weights = [None, [0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.4, 0.4, 0.2],
                    [0.5, 0.4, 0.1], [0.4, 0.5, 0.1], [0.25, 0.25, 0.5]]

    with open(os.path.join(config.GRID_SEARCH_CLF_DIR, 'late_fusion_DND.csv'),'w') as outfile:
        outfile.write('A_wt' + ',' + 'V_wt' + ',' +  'L_wt' + ',' + 'f1_score' + '\n')
        for mode_wt in mode_weights:
            lf_clf = LateFusionClassifier(classifiers=[clf_A, clf_V, clf_L], weights=mode_wt)
            lf_clf.fit(Xs_train, ys_train)
            f1_score = lf_clf.score(Xs_val,y_true_val,scoring='f1')
            if not mode_wt:
                mode_wt = [0.3, 0.3, 0.3]
            outfile.write(str(mode_wt[0]) + ',' + str(mode_wt[1]) + ',' +
                            str(mode_wt[2]) + ',' +str(f1_score) + '\n')
            print f1_score


def main():
    #print "Selecting features...\n"
    #feature_select.feature_select("C")
    #print "Normalizing features...\n"
    #normalize_features()
    norm = 'normalize'
    #print "Performing Grid Search for visual...\n"
    #grid_search_meta(mode='visual', category='PN', normalize=norm)
    #print "Performing Grid Search for acoustic...\n"
    #grid_search_meta(mode='acoustic', category='PN', normalize=norm)
    #print "Performing Grid Search for linguistic...\n"
    #grid_search_meta(mode='linguistic', category='PN', normalize=norm)
    print "Performing Grid Search for Late Fusion...\n"
    grid_search_late_fusion(category='PN',normalize=norm)


if __name__ == '__main__':
    main()
