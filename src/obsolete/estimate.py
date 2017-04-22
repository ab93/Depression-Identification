import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from src.models.regressor import MetaRegressor, LateFusionRegressor
from sklearn.pipeline import _name_estimators
from src.feature_extract.read_labels import features
import src.main.config
import src.main.feature_select
from src.main.utils import get_multi_data, get_single_mode_data
from src.feature_extract import read_labels
from src.helpers.normalized_features import normalize_features
from copy import deepcopy

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
    reg_weights = [None, [0.7,0.3], [0.3,0.7]]

    # Ridge parameters
    r_alphas = np.logspace(-4,4,10)

    # Lasso parameters
    l_alphas = np.logspace(-4,4,5)
    min_mae = 0.0
    max_reg1 = None
    max_reg2 = None
    max_weights = []
    with open(os.path.join(src.main.config.GRID_SEARCH_REG_DIR, mode + '_' + category + '.csv'), 'w') as outfile:
        for reg_wt in reg_weights:
            temp = []
            for i in reg_wt:
                temp.append(float(i))
            for alpha_1 in l_alphas:
                for alpha_2 in l_alphas:
                    for is_normalized in [True, False]:
                        #reg_1 = Ridge(alpha=alpha_1, normalize=is_normalized)
                        #reg_2 = Ridge(alpha=alpha_2, normalize=is_normalized)
                        reg_1 = Lasso(alpha=alpha_1, normalize=is_normalized)
                        reg_2 = Lasso(alpha=alpha_2, normalize=is_normalized)
                        meta_reg = MetaRegressor(regressors=[reg_1, reg_2], weights=reg_wt)
                        meta_reg.fit(X_train, y_train)
                        #r2_score = meta_reg.score(X_val, y_true_val)
                        mean_abs_error = meta_reg.score(X_val, y_true_val, scoring='mean_abs_error')
                        if mean_abs_error < min_mae:
                            min_mae = mean_abs_error
                            max_reg1 = reg_1
                            max_reg2 = reg_2
                            max_weights = temp[:]

                        if not reg_wt:
                            reg_wt = [0.5, 0.5]
                        outfile.write(str(reg_wt[0]) + ' ' + str(reg_wt[1]) +
                                    ',' + str(is_normalized) + ',' + str(alpha_1) +
                                    ',' + str(alpha_2) + ',' + str(r2_score) +',' +
                                    str(mean_abs_error) + '\n')
                        print r2_score, mean_abs_error


def main():
    #print "Selecting features...\n"
    #feature_select.feature_select("R")
    #print "Normalizing features...\n"
    #normalize_features()
    print "Performing Grid Search for visual...\n"
    grid_search_meta(mode='visual', category='PN')
    print "Performing Grid Search for acoustic...\n"
    grid_search_meta(mode='acoustic', category='PN')
    print "Performing Grid Search for linguistic...\n"
    grid_search_meta(mode='linguistic', category='PN')
    #print "Performing Grid Search for Late Fusion...\n"
    #grid_search_lf(category='PN')

if __name__ == '__main__':
    main()
