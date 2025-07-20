import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse
import helper
ONE_KB = 1000

def get_result_df(outDir, intronidx_to_vary:int=1):
    resultFiles = glob.glob(outDir + f'/varyIntron_F{intronidx_to_vary}*_S*.csv')
    resultDfs = []
    timepoints=3
    for resultFile in resultFiles:
        resultDf = pd.read_csv(resultFile, sep=',', header=0, index_col=None)
        for i in range(timepoints):
            resultDf[f'H{i}'] = resultDf[f'before_H{i}'] / resultDf[f'after_H{i}']
            resultDf[f'I{i}'] = resultDf[f'after_I{i}'] / resultDf[f'before_I{i}']
        resultDfs.append(resultDf)
    resultDf = pd.concat(resultDfs, ignore_index=True)
    return resultDf

