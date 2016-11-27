import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.models.classifier import MetaClassifier, LateFusionClassifier
from src.feature_extract import read_labels
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def get_classifiers():
    clf1 = LogisticRegression(C=1, penalty='l2', n_jobs=-1, class_weight={1:4})
    clf2 = LogisticRegression(C=1, penalty='l2', n_jobs=-1, class_weight={1:4})
    #clf1 = SVC(kernel='rbf',degree=2, probability=True, class_weight={1:4})
    #clf2 = SVC(kernel='rbf', probability=True, class_weight={1:4})
    #clf1 = RandomForestClassifier(n_jobs=-1, class_weight={1:2})
    #clf2 = RandomForestClassifier(n_jobs=-1, class_weight={1:2})
    #clf1 = DecisionTreeClassifier()
    #clf2 = DecisionTreeClassifier()
    return [clf1, clf2]

def late_fusion_classify():
    lin_data = read_labels.return_lin_pn([['word76','word87',
                                    '50cogproc_(Cognitive_Processes)',
                                    '31posemo_(Positive_Emotions)']])
    acou_data = read_labels.return_acou_dnd([[ 'MCEP_11','F0', 'HMPDM_10','HMPDM_9','HMPDD_9','HMPDD_11'],
                                            []])
    vis_data = read_labels.return_vis_dnd([['x2','x3','x4','x5','x6'], 
                                            ['Z9','Z54','Z64','Z10'], 
                                            [], 
                                            ['Rx','Ry','Tz'], 
                                            ['AU17Evidence']])
                                          
    X_A_train = [map(np.asarray, acou_data[0]), map(np.asarray, acou_data[2])]
    y_A_train = [map(np.asarray, acou_data[1]), map(np.asarray, acou_data[3])]
    X_A_val = [map(np.asarray, acou_data[4]), map(np.asarray, acou_data[6])]
    y_A_val = [map(np.asarray, acou_data[5]), map(np.asarray, acou_data[7])]

    X_V_train = [map(np.asarray, vis_data[0]), map(np.asarray, vis_data[2])]
    y_V_train = [map(np.asarray, vis_data[1]), map(np.asarray, vis_data[3])]
    X_V_val = [map(np.asarray, vis_data[4]), map(np.asarray, vis_data[6])]
    y_V_val = [map(np.asarray, vis_data[5]), map(np.asarray, vis_data[7])]

    X_L_train = [map(np.asarray, lin_data[0]), map(np.asarray, lin_data[2])]
    y_L_train = [map(np.asarray, lin_data[1]), map(np.asarray, lin_data[3])]
    X_L_val = [map(np.asarray, lin_data[4]), map(np.asarray, lin_data[6])]
    y_L_val = [map(np.asarray, lin_data[5]), map(np.asarray, lin_data[7])]

    Xs = [X_A_train, X_V_train, X_L_train]
    ys = [y_A_train, y_V_train, y_L_train]
    Xs_val = [X_A_val, X_V_val, X_L_val]
    ys_val = [y_A_val, y_V_val, y_L_val]
    
    clf_A = MetaClassifier(classifiers=get_classifiers())
    clf_V = MetaClassifier(classifiers=get_classifiers())
    clf_L = MetaClassifier(classifiers=get_classifiers())
    
    lf_clf = LateFusionClassifier(classifiers=[clf_A, clf_V, clf_L])
    lf_clf.fit(Xs, ys)
    print lf_clf.predict(Xs_val)
    preds = lf_clf.predict_proba(Xs_val)
    y_true = map(int,map(np.mean,y_A_val[0]))
    print lf_clf.score(Xs_val,y_true,scoring='f1')
    for i in xrange(len(y_true)):
        print preds[0][i], preds[1][i], preds[2][i], y_true[i]


def train_classify():
    data = read_labels.return_lin_pn([['word76','word87',
                                    '50cogproc_(Cognitive_Processes)',
                                    '31posemo_(Positive_Emotions)']])

    #data = read_labels.return_acou_dnd([[ 'MCEP_11','F0', 'HMPDM_10','HMPDM_9','HMPDD_9','HMPDD_11'],
    #                                    []])
    X_A_train = [map(np.asarray, data[0]), map(np.asarray, data[2])]
    y_A_train = [map(np.asarray, data[1]), map(np.asarray, data[3])]
    #X_A_train = [map(np.asarray, train_data[4]), map(np.asarray, train_data[6])]
    #y_A_train = [map(np.asarray, train_data[5]), map(np.asarray, train_data[7])]

    X_A_val = [map(np.asarray, data[4]), map(np.asarray, data[6])]
    y_A_val = [map(np.asarray, data[5]), map(np.asarray, data[7])]
    #X_A_val = [map(np.asarray, val_data[4]), map(np.asarray, val_data[6])]
    #y_A_val = [map(np.asarray, val_data[5]), map(np.asarray, val_data[7])]

    print len(X_A_train[0]), len(X_A_train[1])
    print len(y_A_train[0]), len(y_A_train[1])
    raw_input()

    clfs = get_classifiers()
    meta_clf = MetaClassifier(classifiers=clfs)
    meta_clf.fit(X_A_train, y_A_train)

    print len(X_A_train[0]), len(X_A_train[1])
    print len(X_A_val[0]), len(X_A_val[1])
    #raw_input()

    print "\nTraining data...\n"
    
    preds = meta_clf.predict_proba(X_A_train) 
    y_true = map(int,map(np.mean,y_A_train[0]))
    print "F1_score:",meta_clf.score(X_A_train, y_true, scoring='f1')
    print "Accuracy:",meta_clf.score(X_A_train, y_true, scoring='accuracy')

    for i in xrange(len(y_true)):
        print preds[0][i], preds[1][i], y_true[i]

    print "\nTesting data...\n"
    preds = meta_clf.predict_proba(X_A_val) 
    #print meta_clf.predict(X_A_val) 
    y_true = map(int,map(np.mean,y_A_val[0]))
    print meta_clf.score(X_A_val, y_true)
    print meta_clf.score(X_A_val, y_true, scoring='accuracy')
    for i in xrange(len(y_true)):
        print preds[0][i], preds[1][i], y_true[i]
    

def main():
    #train_classify()
    late_fusion_classify()

if __name__ == '__main__':
    main()