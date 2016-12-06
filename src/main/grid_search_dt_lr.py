import config
import os
import pandas


def call_logistic_model(filename):
    classify = pd.read_csv(config.RESULTS_CLASSIFY + "/" + filename+"_PN.csv")
    data = classify.iloc[classify['F1_score'].argmax()]

    return logistic_model(data['class_wt'], data['clf_wt'], data['C1'], data['C2'], data['P1'], data['P2'])