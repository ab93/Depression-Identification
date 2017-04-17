from sklearn.externals import joblib
import utils
import numpy as np
from ..feature_extract import  read_labels
import config
import os

def final_classifier(mode,category="PN",problem_type="C",normalize="normalize"):
    if category == "PN":
        cat_1 = "positive"
        cat_2 = "negative"
    if mode == "late_fusion":
        X_test = [    [   map(np.asarray, read_labels.features("acoustic", cat_1, "test", problem_type, normalize)),
                      map(np.asarray, read_labels.features("acoustic", cat_2, "test", problem_type, normalize))
                  ],
                  [   map(np.asarray, read_labels.features("visual", cat_1, "test", problem_type, normalize)),
                      map(np.asarray, read_labels.features("visual", cat_2, "test", problem_type, normalize))
                  ],
                  [   map(np.asarray, read_labels.features("linguistic", cat_1, "test", problem_type, normalize)),
                      map(np.asarray, read_labels.features("linguistic", cat_2, "test", problem_type, normalize))
                  ]
              ]
    else:

        X_test = [map(np.asarray, read_labels.features(mode, cat_1, "test", problem_type, normalize)),
             map(np.asarray, read_labels.features(mode, cat_2, "test", problem_type, normalize))]

    clf = joblib.load(os.path.join(config.GRID_SEARCH_CLF_DIR, mode + '_pickle' + category + '.pkl'))
    preds_label = clf.predict(X_test)
    return preds_label

def final_estimator(mode,category="PN",problem_type="R",normalize="normalize"):
    if category == "PN":
        cat_1 = "positive"
        cat_2 = "negative"
    if mode == "late_fusion":
        X_test = [    [   map(np.asarray, read_labels.features("acoustic", cat_1, "test", problem_type, normalize)),
                      map(np.asarray, read_labels.features("acoustic", cat_2, "test", problem_type, normalize))
                  ],
                  [   map(np.asarray, read_labels.features("visual", cat_1, "test", problem_type, normalize)),
                      map(np.asarray, read_labels.features("visual", cat_2, "test", problem_type, normalize))
                  ],
                  [   map(np.asarray, read_labels.features("linguistic", cat_1, "test", problem_type, normalize)),
                      map(np.asarray, read_labels.features("linguistic", cat_2, "test", problem_type, normalize))
                  ]
              ]
    else:

        X_test = [map(np.asarray, read_labels.features(mode, cat_1, "test", problem_type, normalize)),
             map(np.asarray, read_labels.features(mode, cat_2, "test", problem_type, normalize))]

    reg = joblib.load(os.path.join(config.GRID_SEARCH_REG_DIR, mode + '_pickle' + category + '.pkl'))
    preds_label = reg.predict(X_test)
    return preds_label

def main_classifier():
    print "acoustic"
    print final_classifier("acoustic")
    print "visual"
    print final_classifier("visual")
    print "linguistic"
    print final_classifier("linguistic")
    print "late_fusion"
    print final_classifier("late_fusion")

def main_regressor():
    print "acoustic"
    print final_estimator("acoustic")
    print "visual"
    print final_estimator("visual")
    print "linguistic"
    print final_estimator("linguistic")
    print "late_fusion"
    print final_estimator("late_fusion")

#main_classifier()
main_regressor()

