import os
import sys
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from src.models.regressor import MetaRegressor, LateFusionRegressor
from sklearn.pipeline import _name_estimators
from ..feature_extract.read_labels import features
import config
import feature_select
from utils import get_multi_data, get_single_mode_data
from src.feature_extract import read_labels
from ..helpers.normalized_features import normalize_features
from copy import deepcopy
from sklearn.tree import DecisionTreeRegressor


def grid_search_meta(mode='acoustic',category='PN'):
    X_train, y_train, X_val, y_val = get_single_mode_data(mode, category,
                                    problem_type='R')
    X_data = deepcopy(X_train)
    X_data[0].extend(X_val[0])
    X_data[1].extend(X_val[1])
    y_data = deepcopy(y_train)
    y_data[0].extend(y_val[0])
    y_data[1].extend(y_val[1])

    # Set y_true for validation
    y_true_val = map(int,map(np.mean,y_val[0]))

    # Set parameters for GridSearch
    reg_weights = [[0.7,0.3], [0.3,0.7]]

    # Ridge parameters
    r_alphas = np.logspace(-4,4,10)

    # Lasso parameters
    l_alphas = np.logspace(-4,4,5)
    min_mae = float(sys.maxint)
    max_reg1 = None
    max_reg2 = None
    max_weights = []
    # with open(os.path.join(config.GRID_SEARCH_REG_DIR, mode + '_' + category + '.csv'),'w') as outfile:
    #
    #     for reg_wt in reg_weights:
    #         temp = []
    #         for i in reg_wt:
    #             temp.append(float(i))
    #         for alpha_1 in l_alphas:
    #             for alpha_2 in l_alphas:
    #                 for is_normalized in [True, False]:
    #                     #reg_1 = Ridge(alpha=alpha_1, normalize=is_normalized)
    #                     #reg_2 = Ridge(alpha=alpha_2, normalize=is_normalized)
    #                     reg_1 = Lasso(alpha=alpha_1, normalize=is_normalized)
    #                     reg_2 = Lasso(alpha=alpha_2, normalize=is_normalized)
    #                     meta_reg = MetaRegressor(regressors=[reg_1, reg_2], weights=reg_wt)
    #                     meta_reg.fit(X_train, y_train)
    #                     #r2_score = meta_reg.score(X_val, y_true_val)
    #                     mean_abs_error = meta_reg.score(X_val, y_true_val, scoring='mean_abs_error')
    #                     if mean_abs_error < min_mae:
    #                         min_mae = mean_abs_error
    #                         max_reg1 = reg_1
    #                         max_reg2 = reg_2
    #                         max_weights = temp[:]
    #
    #                     if not reg_wt:
    #                         reg_wt = [0.5, 0.5]
    #                     outfile.write(str(reg_wt[0]) + ' ' + str(reg_wt[1]) +
    #                                 ',' + str(is_normalized) + ',' + str(alpha_1) +
    #                                 ',' + str(alpha_2) + ',' + str(r2_score) +',' +
    #                                 str(mean_abs_error) + '\n')
    #                     print r2_score, mean_abs_error
    max_features_ = np.arange(3, 20, 5)
    max_depths = np.arange(3, 6)
    min_samples_leaves = np.arange(2, 6)
    reg1_params = [{'alpha': x, 'normalize': y}
               for x in l_alphas for y in [True,False]]
    reg2_params = [{'max_features': x, 'max_depth': y, 'min_samples_leaf': z}
               for x in max_features_ for y in max_depths for z in min_samples_leaves]
    reg_params = [reg1_params,reg2_params]
    estimators = [Lasso,DecisionTreeRegressor]

    for reg_wt in reg_weights:
        temp = []
        for i in reg_wt:
            temp.append(float(i))
        for i,reg1 in enumerate(estimators):
            for j,reg2 in enumerate(estimators):
                for param1 in reg_params[i]:
                    for param2 in reg_params[j]:
                        reg_1 = reg1(**param1)
                        reg_2 = reg2(**param2)
                        meta_reg = MetaRegressor(regressors=[reg_1, reg_2], weights=temp)
                        meta_reg.fit(X_train, y_train)
                        mean_abs_error = meta_reg.score(X_val, y_true_val, scoring='mean_abs_error')
                        if mean_abs_error < min_mae:
                            min_mae = mean_abs_error
                            max_reg1 = reg_1
                            max_reg2 = reg_2
                            max_weights = temp[:]
                            best_val = mean_abs_error

                        print i,j,mean_abs_error,temp

    print best_val
    meta_reg = MetaRegressor(regressors=[max_reg1, max_reg2], weights=max_weights)
    meta_reg.fit(X_data, y_data)
    joblib.dump(meta_reg, os.path.join(config.GRID_SEARCH_REG_DIR, mode + '_pickle' + category + '.pkl'))



def grid_search_late_fusion(category='PN', normalize='normalize'):
    Xs_train, ys_train, Xs_val, ys_val = get_multi_data(category=category,problem_type="R", normalize=normalize)
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

    reg_a =  joblib.load(os.path.join(config.GRID_SEARCH_REG_DIR  + '/acoustic_pickle' + category + '.pkl'))
    reg_v =  joblib.load(os.path.join(config.GRID_SEARCH_REG_DIR  + '/visual_pickle' + category + '.pkl'))
    reg_l = joblib.load(os.path.join(config.GRID_SEARCH_REG_DIR + '/linguistic_pickle' + category + '.pkl'))
    min_mae = float(sys.maxint)
    max_reg = None
    mode_weights = [None, [0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.4, 0.4, 0.2],
                    [0.5, 0.4, 0.1], [0.4, 0.5, 0.1], [0.25, 0.25, 0.5]]
    for mode_wt in mode_weights:
        lf_reg = LateFusionRegressor(regressors=[reg_a, reg_v, reg_l], weights=mode_wt)
        lf_reg.fit(Xs_train, ys_train)
        mae_score = lf_reg.score(Xs_val, y_true_val, scoring='mean_abs_error')
        if not mode_wt:
            mode_wt = [0.3, 0.3, 0.3]
        #print mae_score
        if(mae_score < min_mae):
            min_mae = mae_score
            max_reg = lf_reg
            best_val = min_mae
    print best_val

    max_reg.fit(X_data, y_data)
    joblib.dump(max_reg, os.path.join(config.GRID_SEARCH_REG_DIR + '/late_fusion_picklePN.pkl'))


def main():
    #print "Selecting features...\n"
    #feature_select.feature_select("R")
    #print "Normalizing features...\n"
    #normalize_features()
    #print "Performing Grid Search for visual...\n"
    #grid_search_meta(mode='visual', category='PN')
    #print "Performing Grid Search for acoustic...\n"
    #grid_search_meta(mode='acoustic', category='PN')
    #print "Performing Grid Search for linguistic...\n"
    #grid_search_meta(mode='linguistic', category='PN')
    print "Performing Grid Search for Late Fusion...\n"
    grid_search_late_fusion(category='PN')

if __name__ == '__main__':
    main()
