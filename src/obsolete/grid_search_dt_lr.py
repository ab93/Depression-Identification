import src.main.config
import os
import pandas as pd
import numpy as np
from pprint import pprint


def main(mode,classifier):
    if classifier == "DT":
        file_type = ".txt"
        seperator = "\t"
        column = 4
    else:
        file_type = ".csv"
        seperator = ","
        column = 6
    data = pd.read_csv(src.main.config.GRID_SEARCH_CLF_DIR + "/" + mode + "_" + classifier + "_PN" + file_type, sep=seperator, header= None)
    result = data.sort(column,ascending=False)
    result = result.iloc[0:5]
    #print data
   # print result.unique()
    return result

def write():
    result_acoustic_LR = main("acoustic", "LR")
    result_visual_LR = main("visual", "LR")
    result_ling_LR = main("linguistic", "LR")
    result_LR = result_acoustic_LR.append(result_visual_LR)
    result_LR = result_LR.append(result_ling_LR)
    #print result_LR
    result_LR.to_csv(src.main.config.GRID_SEARCH_CLF_DIR + "/" + "refined_LR.csv", index=None)
    result_acoustic_DT = main("acoustic", "DT")
    result_visual_DT = main("visual", "DT")
    result_ling_DT = main("linguistic", "DT")
    result_DT = result_acoustic_DT.append(result_visual_DT)
    result_DT = result_DT.append(result_ling_DT)
    result_DT.to_csv(src.main.config.GRID_SEARCH_CLF_DIR + "/" + "refined_DT.csv", index=None)
    res_LR = {}
    res_DT = {}
    for i in range(result_LR.shape[1]-1):
        res_LR[i] = result_LR[i].unique()
    for i in range(result_DT.shape[1]-1):
        res_DT[i] = result_DT[i].unique()
    #pprint(res_LR)
    #pprint(res_DT)
    return res_LR,res_DT

def ret_func():
    res_LR,res_DT = write()
    res_LR[0] = list(set(res_LR[0]).union(set(res_DT[0])))
    del res_DT[0]
    res_LR[1] = list(set(res_LR[1]).union(set(res_DT[3])))
    del res_DT[3]
    res_LR[2] = list(set(res_LR[2]).union(set(res_LR[3])))
    del res_LR[3]

    res_LR[4] = list(set(res_LR[4]).union(set(res_LR[5])))
    del res_LR[5]

    res_DT[1] = list(set(res_DT[1]).union(set(res_DT[2])))
    del res_DT[2]

    #pprint(res_LR)
    #pprint(res_DT)
    return res_LR,res_DT

#ret_func()

