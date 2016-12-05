import utils
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import os
import numpy as np
import config
import pandas as pd
import utils
from sklearn.linear_model import LogisticRegression
from ..models import classifier
from sklearn.metrics import roc_auc_score

def logistic_model(intra_clf_weights,inter_clf_weights,c1,c2,p1,p2):
    inter_clf_weights = inter_clf_weights.split(" ")
    temp = []
    for i in inter_clf_weights:
        temp.append(float(i))
    X_train, y_train, X_val, y_val = utils.get_single_mode_data()
    y_true = map(int, map(np.mean, y_val[0]))

    clf1 = LogisticRegression(n_jobs=-1, class_weight={1: intra_clf_weights},C=c1,penalty=p1)
    clf2 = LogisticRegression(n_jobs=-1, class_weight={1: intra_clf_weights},C=c2,penalty=p2)
    meta_clf = classifier.MetaClassifier(classifiers=[clf1,clf2], weights=temp)
    meta_clf.fit(X_train, y_train)

    preds = meta_clf.predict_proba(X_val, get_all=False)
    preds_positive_labels = []
    for i in preds:
        preds_positive_labels.append(i[1])
    f1 = meta_clf.score(X_val, y_true)
    accuracy = meta_clf.score(X_val, y_true, scoring='accuracy')
    fpr, tpr, thresholds = roc_curve(y_true, preds_positive_labels)
    roc_area = roc_auc_score(y_true, preds_positive_labels)
    return clf1,clf2,f1,accuracy,fpr,tpr,thresholds,roc_area

def call_logistic_model(filename):
    classify = pd.read_csv(config.RESULTS_CLASSIFY + "/" + filename+"_PN.csv")
    data = classify.iloc[classify['F1_score'].argmax()]
    if filename == 'visual':
        print data
        raw_input()
    return logistic_model(data['class_wt'], data['clf_wt'], data['C1'], data['C2'], data['P1'], data['P2'])

def plot_roc_curve(fpr_a,tpr_a,roc_area_a,fpr_v,tpr_v,roc_area_v,fpr_l,tpr_l,roc_area_l):

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot(fpr_a, tpr_a,lw=2,label='ROC Acoustic (area = %0.2f)' % roc_area_a)
    plt.plot(fpr_v, tpr_v,lw=2,label='ROC Visual (area = %0.2f)' % roc_area_v)
    plt.plot(fpr_l, tpr_l,lw=2,label='ROC Linguistic (area = %0.2f)' % roc_area_l)
    plt.legend(loc="best")
    plt.show()

def plot_roc_latefusion(fpr,tpr,roc_area):
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot(fpr, tpr, lw=2, label='ROC Late Fusion(area = %0.2f)' % roc_area)
    plt.legend(loc="best")
    plt.show()

def late_fusion_model(clf1_a,clf2_a,clf1_v,clf2_v,clf1_l,clf2_l):
    # Read the data
    Xs_train, ys_train, Xs_val, ys_val = utils.get_multi_data()
    clf_A = classifier.MetaClassifier(classifiers=[clf1_a, clf2_a])
    clf_V = classifier.MetaClassifier(classifiers=[clf1_v, clf2_v])
    clf_L = classifier.MetaClassifier(classifiers=[clf1_l, clf2_l])
    lf_clf = classifier.LateFusionClassifier(classifiers=[clf_A, clf_V, clf_L], weights=[0.4, 0.4, 0.2])
    lf_clf.fit(Xs_train, ys_train)
    y_true = map(int, map(np.mean, ys_val[0][0]))
    f1 = lf_clf.score(Xs_val, y_true, scoring='f1')
    accuracy = lf_clf.score(Xs_val, y_true, scoring='accuracy')
    preds = lf_clf.predict_proba(Xs_val, get_all=False)
    preds_positive_labels = []
    for i in preds:
        preds_positive_labels.append(i[1])



    fpr, tpr, thresholds = roc_curve(y_true, preds_positive_labels)
    roc_area = roc_auc_score(y_true, preds_positive_labels)
    plot_roc_latefusion(fpr,tpr,roc_area)

    print "F1 score: ",f1
    print "Accuracy: ",accuracy
    print "ROC Area: ",roc_area




def main():

    print "Acoustic"
    clf1_a,clf2_a,f1_a,accuracy_a,fpr_a,tpr_a,thresholds_a,roc_area_a = call_logistic_model('acoustic')
    print "F1 score: ",f1_a
    print "Accuracy: ",accuracy_a
    print "ROC Area: ",roc_area_a
    print "\n"
    print "Visual"
    clf1_v, clf2_v,f1_v, accuracy_v, fpr_v, tpr_v, thresholds_v, roc_area_v = call_logistic_model('visual')
    print "F1 score: ",f1_v
    print "Accuracy: ",accuracy_v
    print "ROC Area: ",roc_area_v
    print "\n"
    print "linguistic"
    clf1_l, clf2_l,f1_l, accuracy_l, fpr_l, tpr_l, thresholds_l, roc_area_l = call_logistic_model('linguistic')
    print "F1 score: ",f1_l
    print "Accuracy: ",accuracy_l
    print "ROC Area: ",roc_area_l
    print "\n"
    print "Late Fusion"
    #plot_roc_curve(fpr_a,tpr_a,roc_area_a,fpr_v,tpr_v,roc_area_v,fpr_l,tpr_l,roc_area_l)
    late_fusion_model(clf1_a,clf2_a,clf1_v,clf2_v,clf1_l,clf2_l)

#main()
classify = pd.read_csv(config.RESULTS_CLASSIFY + "/visual_PN.csv")
data = classify.iloc[classify['F1_score'].argmax()]
print data
print logistic_model(data['class_wt'], data['clf_wt'], data['C1'], data['C2'], data['P1'], data['P2'])