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
from sklearn.naive_bayes import GaussianNB
import config

def oversample(X,y):
    X = np.vstack(X)
    y = np.hstack(y)
    df = pd.DataFrame(X)
    df['labels'] = y
    grouped_df = df.groupby('labels')
    for key, dframe in grouped_df:
        if key == 1:
            sampled_df = dframe
    df = df.append(sampled_df)
    data = df.values
    X, y = data[:,:-1], data[:,-1]
    return X, y
        
    
def get_classifiers():
    clf1 = LogisticRegression(C=1, penalty='l1', n_jobs=-1, class_weight={1:4})
    clf2 = LogisticRegression(C=1, penalty='l1', n_jobs=-1, class_weight={1:1.5})
    #clf1 = SVC(kernel='rbf',degree=2, probability=True, class_weight={1:1})
    #clf2 = SVC(kernel='rbf', probability=True, class_weight={1:1})
    #clf1 = RandomForestClassifier(n_jobs=-1, class_weight={1:2})
    #clf2 = RandomForestClassifier(n_jobs=-1, class_weight={1:2})
    #clf1 = DecisionTreeClassifier()
    #clf2 = DecisionTreeClassifier()
    #clf1 = GaussianNB()
    #clf2 = GaussianNB()
    return [clf1, clf2]


def get_data():
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

    # Set the data            
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

    return Xs, ys, Xs_val, ys_val


def late_fusion_classify():
    # Read the data
    Xs_train, ys_train, Xs_val, ys_val = get_data()
    
    clf_A_D = LogisticRegression(C=1.0, penalty='l2', class_weight={1:4})
    clf_A_ND = LogisticRegression(C=1.0, penalty='l2', class_weight={1:4})

    clf_V_D = LogisticRegression(C=1.0, penalty='l2', class_weight={1:3})
    clf_V_ND = LogisticRegression(C=1.0, penalty='l2', class_weight={1:3})

    clf_L_D = LogisticRegression(C=1.0, penalty='l2', class_weight={1:3})
    clf_L_ND = LogisticRegression(C=1.0, penalty='l2', class_weight={1:3})

    clf_A = MetaClassifier(classifiers=[clf_A_D, clf_A_ND])
    clf_V = MetaClassifier(classifiers=[clf_V_D, clf_V_ND])
    clf_L = MetaClassifier(classifiers=[clf_L_D, clf_L_ND])
    
    lf_clf = LateFusionClassifier(classifiers=[clf_A, clf_V, clf_L], weights=[0.6,0.2,0.1])
    lf_clf.fit(Xs_train, ys_train)
    print lf_clf.predict(Xs_val)
    preds = lf_clf.predict_proba(Xs_val, get_all=True)
    y_true = map(int,map(np.mean,ys_val[0][0]))
    print lf_clf.score(Xs_val,y_true,scoring='f1')
    for i in xrange(len(y_true)):
        print preds[0][i], preds[1][i], preds[2][i], y_true[i]


def grid_search_lf():
    Xs_train, ys_train, Xs_val, ys_val = get_data()
    y_true_val = map(int,map(np.mean,ys_val[0][0]))

    clf_A_D = LogisticRegression(C=1.0, penalty='l2', class_weight={1:4})
    clf_A_ND = LogisticRegression(C=1.0, penalty='l2', class_weight={1:4})

    clf_V_D = LogisticRegression(C=1.0, penalty='l2', class_weight={1:3})
    clf_V_ND = LogisticRegression(C=1.0, penalty='l2', class_weight={1:3})

    clf_L_D = LogisticRegression(C=1.0, penalty='l2', class_weight={1:3})
    clf_L_ND = LogisticRegression(C=1.0, penalty='l2', class_weight={1:3})

    clf_A = MetaClassifier(classifiers=[clf_A_D, clf_A_ND])
    clf_V = MetaClassifier(classifiers=[clf_V_D, clf_V_ND])
    clf_L = MetaClassifier(classifiers=[clf_L_D, clf_L_ND])

    mode_weights = [None, [0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.4, 0.4, 0.2]]

    with open(os.path.join(config.GRID_SEARCH_DIR, 'late_fusion.csv'),'w') as outfile:
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



def grid_search_meta(mode='acoustic'):
    # Read data
    if mode == 'acoustic':
        data = read_labels.return_acou_dnd([[ 'MCEP_11','F0', 'HMPDM_10','HMPDM_9','HMPDD_9','HMPDD_11'],
                                            []])
    elif mode == 'visual':
        data = read_labels.return_vis_dnd([['x2','x3','x4','x5','x6'], 
                                            ['Z9','Z54','Z64','Z10'], 
                                            [], 
                                            ['Rx','Ry','Tz'], 
                                            ['AU17Evidence']])
    else:
        data = read_labels.return_lin_pn([['word76','word87',
                                    '50cogproc_(Cognitive_Processes)',
                                    '31posemo_(Positive_Emotions)']])

    X_train = [map(np.asarray, data[0]), map(np.asarray, data[2])]
    y_train = [map(np.asarray, data[1]), map(np.asarray, data[3])]
    X_val = [map(np.asarray, data[4]), map(np.asarray, data[6])]
    y_val = [map(np.asarray, data[5]), map(np.asarray, data[7])]

    # Set y_true for validation
    y_true_val = map(int,map(np.mean,y_val[0]))

    # Set parameters for GridSearch
    class_weights = np.arange(1,5)
    C_vals = np.logspace(-3,3,num=5)
    gammas = np.logspace(-3,3,num=5)
    penalties = ('l1','l2')
    clf_weights = [None, [0.7,0.3], [0.3,0.7]]

    results = {}
    with open(os.path.join(config.GRID_SEARCH_DIR, mode + '.csv'),'w') as outfile:
        for p1 in penalties:
            for p2 in penalties:
                for clf_wt in clf_weights:
                    for class_wt in class_weights:
                        for C1 in C_vals:
                            for C2 in C_vals: 
                            #for gamma in gammas:
                                clf_D = LogisticRegression(C=C1, penalty=p1, n_jobs=-1, class_weight={1:class_wt})
                                clf_ND = LogisticRegression(C=C2, penalty=p2, n_jobs=-1, class_weight={1:class_wt})
                                meta_clf = MetaClassifier(classifiers=[clf_D, clf_ND], weights=clf_wt)
                                meta_clf.fit(X_train, y_train)
                                f1_score = meta_clf.score(X_val, y_true_val)
                                if not clf_wt:
                                    clf_wt = [0.5, 0.5]
                                outfile.write(str(clf_wt[0]) + ' ' + str(clf_wt[1]) + ',' + 
                                            str(class_wt) + ',' + str(C1) + ',' + str(C2) + ',' 
                                            + p1 + ',' + p2 + ','
                                            + str(f1_score) + '\n')
                                print f1_score



def train_classify():
    data = read_labels.return_lin_pn([['word76','word87',
                                    '50cogproc_(Cognitive_Processes)',
                                    '31posemo_(Positive_Emotions)']])

    #data = read_labels.return_acou_dnd([[ 'MCEP_11','F0', 'HMPDM_10','HMPDM_9','HMPDD_9','HMPDD_11'],
    #                                    []])
    X_A_train = [map(np.asarray, data[0]), map(np.asarray, data[2])]
    y_A_train = [map(np.asarray, data[1]), map(np.asarray, data[3])]
    
    # X_D, y_D = oversample(X_A_train[0], y_A_train[0])
    # X_ND, y_ND = oversample(X_A_train[1], y_A_train[1])
    # X_A_train = [X_D, X_ND]
    # y_A_train = [y_D, y_ND]

    X_A_val = [map(np.asarray, data[4]), map(np.asarray, data[6])]
    y_A_val = [map(np.asarray, data[5]), map(np.asarray, data[7])]
    #X_A_val = [map(np.asarray, val_data[4]), map(np.asarray, val_data[6])]
    #y_A_val = [map(np.asarray, val_data[5]), map(np.asarray, val_data[7])]

    print len(X_A_train[0]), len(X_A_train[1])

    clfs = get_classifiers()
    meta_clf = MetaClassifier(classifiers=clfs, weights=[0.9, 0.1])
    meta_clf.fit(X_A_train, y_A_train)

    print len(X_A_train[0]), len(X_A_train[1])
    print len(X_A_val[0]), len(X_A_val[1])
    #raw_input()

    # print "\nTraining data...\n"
    
    # preds = meta_clf.predict_proba(X_A_train, get_all=True) 
    # y_true = map(int,map(np.mean,y_A_train[0]))
    # print "F1_score:",meta_clf.score(X_A_train, y_true, scoring='f1')
    # print "Accuracy:",meta_clf.score(X_A_train, y_true, scoring='accuracy')

    # for i in xrange(len(y_true)):
    #     print preds[0][i], preds[1][i], y_true[i]

    print "\nTesting data...\n"
    preds = meta_clf.predict_proba(X_A_val, get_all=True) 
    #print meta_clf.predict(X_A_val) 
    y_true = map(int,map(np.mean,y_A_val[0]))
    print meta_clf.score(X_A_val, y_true)
    print meta_clf.score(X_A_val, y_true, scoring='accuracy')
    for i in xrange(len(y_true)):
        print preds[0][i], preds[1][i], y_true[i]
    

def main():
    #train_classify()
    #late_fusion_classify()
    #grid_search_meta(mode='visual')
    #grid_search_meta(mode='acoustic')
    #grid_search_meta(mode='linguistic')
    grid_search_lf()

if __name__ == '__main__':
    main()