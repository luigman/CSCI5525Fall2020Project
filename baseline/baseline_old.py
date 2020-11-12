import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

def baseline():
    mobility_dataframe = pd.read_csv('google_baseline_test.csv', infer_datetime_format=True, parse_dates=True)
    cdc_dataframe = pd.read_csv('cdc_baseline_test.csv', infer_datetime_format=True, parse_dates=True)

    full_dataframe = pd.concat([mobility_dataframe, cdc_dataframe], axis=1)

    sns.pairplot(full_dataframe[['newAndPnew', 'retail_and_recreation_percent_change_from_baseline', 'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline']], diag_kind='kde')
    #plt.show()

    bestWorkCorr = -1
    bestResCorr = 1
    bestRetailCorr = -1
    bestWorkOffset = -1
    bestResOffset = -1
    bestRetailOffset = -1

    for offset in range(60):
        
        cdc_dataframe_truc = cdc_dataframe.shift(periods=offset,fill_value=0)
        mobility_dataframe_truc = mobility_dataframe.drop(columns=['date'])

        full_dataframe = pd.concat([cdc_dataframe_truc, mobility_dataframe_truc], axis=1)
        work = full_dataframe['workplaces_percent_change_from_baseline']
        res = full_dataframe['residential_percent_change_from_baseline']
        retail = full_dataframe['retail_and_recreation_percent_change_from_baseline']
        cases = full_dataframe['newAndPnew']

        correlation = full_dataframe.corr()
        print(correlation.to_numpy())
        print(correlation.to_numpy()[0,1:])
        workCorrelation = cases.corr(work)
        resCorrelation = cases.corr(res)
        retailCorrelation = cases.corr(retail)

        print("Offset:", offset, "Work Correlation:", workCorrelation)
        print("           Res Correlation:", resCorrelation)
        print("           Retail Correlation:", retailCorrelation)

        if workCorrelation > bestWorkCorr:
            bestWorkCorr = workCorrelation
            bestWorkOffset = offset

        if resCorrelation < bestResCorr:
            bestResCorr = resCorrelation
            bestResOffset = offset

        if retailCorrelation > bestRetailCorr:
            bestRetailCorr = retailCorrelation
            bestRetailOffset = offset

    print("Best Work Correlation:", bestWorkCorr)
    print("Best Work Offset:", bestWorkOffset)
    print("Best Res Correlation:", bestResCorr)
    print("Best Res Offset:", bestResOffset)
    print("Best Retail Correlation:", bestRetailCorr)
    print("Best Retail Offset:", bestRetailOffset)

    normalizer = preprocessing.Normalization()
    #normalizer.adapt(np.array(mobility_dataframe))

if __name__ == '__main__':
    baseline()