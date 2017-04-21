import src.main.utils
import itertools
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import os
import numpy as np
import src.main.config
import pandas as pd
import src.main.utils
from sklearn.linear_model import LogisticRegression
from src.models import classifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

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
class_names = ["Non-Depressed","Depressed"]
def late_fusion_model(clf1_a,clf2_a,clf1_v,clf2_v,clf1_l,clf2_l):
    # Read the data
    Xs_train, ys_train, Xs_val, ys_val = src.main.utils.get_multi_data()
    clf_A = classifier.MetaClassifier(classifiers=[clf1_a, clf2_a])
    clf_V = classifier.MetaClassifier(classifiers=[clf1_v, clf2_v])
    clf_L = classifier.MetaClassifier(classifiers=[clf1_l, clf2_l])
    lf_clf = classifier.LateFusionClassifier(classifiers=[clf_A, clf_V, clf_L], weights=[0.4, 0.4, 0.2])
    lf_clf.fit(Xs_train, ys_train)
    y_true = map(int, map(np.mean, ys_val[0][0]))
    f1 = lf_clf.score(Xs_val, y_true, scoring='f1')
    accuracy = lf_clf.score(Xs_val, y_true, scoring='accuracy')
    preds = lf_clf.predict_proba(Xs_val, get_all=False)
    preds_label = lf_clf.predict(Xs_val)
    preds_positive_labels = []
    for i in preds:
        preds_positive_labels.append(i[1])



    fpr, tpr, thresholds = roc_curve(y_true, preds_positive_labels)
    roc_area = roc_auc_score(y_true, preds_positive_labels)
    plot_roc_latefusion(fpr,tpr,roc_area)

    print "F1 score: ",f1
    print "Accuracy: ",accuracy
    print "ROC Area: ",roc_area
    cnf_matrix = confusion_matrix(y_true,preds_label)
    plot_confusion_matrix(cnf_matrix,classes = class_names,title="Confusion Matrix Late Fusion")


def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def logistic_model(mode,inter_clf_weights,clf1,clf2):
    # print clf1
    # print clf2

    inter_clf_weights = eval(inter_clf_weights)
    temp = []
    for i in inter_clf_weights:
        #print i
        temp.append(float(i))
    #print temp

    X_train, y_train, X_val, y_val = src.main.utils.get_single_mode_data(mode=mode, normalize='normalize')
    print X_train
    #print y_train
    #print X_val
    raw_input()

    y_true = map(int, map(np.mean, y_val[0]))
    clf1 = eval(clf1)
    clf2 = eval(clf2)
    meta_clf = classifier.MetaClassifier(classifiers=[clf1,clf2], weights=temp)
    meta_clf.fit(X_train, y_train)

    preds = meta_clf.predict_proba(X_val, get_all=False)
    preds_label = meta_clf.predict(X_val)
    preds_positive_labels = []

    for i in preds:
        preds_positive_labels.append(i[1])

    f1 = meta_clf.score(X_val, y_true)
    accuracy = meta_clf.score(X_val, y_true, scoring='accuracy')
    fpr, tpr, thresholds = roc_curve(y_true, preds_positive_labels)
    roc_area = roc_auc_score(y_true, preds_positive_labels)
    cnf_matrix = confusion_matrix(y_true, preds_label)
    return clf1,clf2,f1,accuracy,fpr,tpr,thresholds,roc_area,cnf_matrix

def call_logistic_model(mode):
    classify = pd.read_csv(src.main.config.RESULTS_CLASSIFY + "/" + mode + "_test_PN.txt", sep="\t", header=None)
    data = classify.iloc[classify[4].argmax()]
    return logistic_model(mode,data[0],data[1],data[3])

def main():

    print "Acoustic"
    clf1_a, clf2_a, f1_a, accuracy_a, fpr_a, tpr_a, thresholds_a, roc_area_a, cnf_matrix_a = call_logistic_model('acoustic')
    print "F1 score: ",f1_a
    print "Accuracy: ",accuracy_a
    print "ROC Area: ",roc_area_a
    print "\n"
    print "Visual"
    clf1_v, clf2_v,f1_v, accuracy_v, fpr_v, tpr_v, thresholds_v, roc_area_v,cnf_matrix_v = call_logistic_model('visual')
    print "F1 score: ",f1_v
    print "Accuracy: ",accuracy_v
    print "ROC Area: ",roc_area_v
    print "\n"
    print "linguistic"
    clf1_l, clf2_l,f1_l, accuracy_l, fpr_l, tpr_l, thresholds_l, roc_area_l,cnf_matrix_l = call_logistic_model('linguistic')
    print "F1 score: ",f1_l
    print "Accuracy: ",accuracy_l
    print "ROC Area: ",roc_area_l
    print "\n"
    print "Late Fusion"

    # plot_roc_curve(fpr_a, tpr_a, roc_area_a, fpr_v, tpr_v, roc_area_v, fpr_l, tpr_l, roc_area_l)
    # plot_confusion_matrix(cnf_matrix_a, classes=class_names,title="Confusion Matrix Acoustic")
    # plot_confusion_matrix(cnf_matrix_v, classes=class_names,title="Confusion Matrix Visual")
    # plot_confusion_matrix(cnf_matrix_l, classes=class_names,title="Confusion Matrix Linguistic")
    late_fusion_model(clf1_a,clf2_a,clf1_v,clf2_v,clf1_l,clf2_l)

main()
