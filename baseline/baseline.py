import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

def baseline():
    showPlot = True
    np.set_printoptions(precision=3, suppress=True)

    mobility_dataframe = pd.read_csv('google_baseline_test.csv', infer_datetime_format=True, parse_dates=True)
    cdc_dataframe = pd.read_csv('cdc_baseline_test_movingAvg.csv', infer_datetime_format=True, parse_dates=True)

    #=========================FIND BEST OFFSET========================================

    bestLinearCorr = 0
    bestLogCorr = 0
    bestLinearOffset = -1
    bestLogOffset = -1
    bestLinearData = 0
    bestLogData = 0

    correlationScores = []
    correlationLogScores = []

    for offset in range(100):
        #Shift CDC data by offset value
        cdc_dataframe_truc = cdc_dataframe.shift(periods=offset,fill_value=0)

        #Build new full data array
        mobility_dataframe_truc = mobility_dataframe.drop(columns=['date'])
        full_dataframe = pd.concat([cdc_dataframe_truc, mobility_dataframe_truc], axis=1)
        full_dataframe['originalCases'] = cdc_dataframe['newAndPnew'] #preserve original case values as additional feature
        full_dataframe_noDate = full_dataframe.drop(columns=['submission_date'])
        full_dataframe_noDate = full_dataframe_noDate.loc[(full_dataframe_noDate['newAndPnew']!=0)] #remove rows with zero cases

        #Compute linear and logatrithmic correlations
        linearCorr = full_dataframe_noDate.corr()
        linearCorr = linearCorr.to_numpy()[0,1:] #Take only correlations between 'cases' and mobility data

        logData = np.log(full_dataframe_noDate+1-np.min(full_dataframe_noDate.to_numpy()))
        logCorr = logData.corr()
        logCorr = logCorr.to_numpy()[0,1:] #Take only correlations between 'cases' and mobility data

        print("Offset:", offset, "Correlation:    ", linearCorr)
        print("           Log Correlation:", logCorr)

        #Save best values
        if np.linalg.norm(linearCorr) > np.linalg.norm(bestLinearCorr):
            bestLinearCorr = linearCorr
            bestLinearOffset = offset
            bestLinearData = full_dataframe_noDate

        if np.linalg.norm(logCorr) > np.linalg.norm(bestLogCorr):
            bestLogCorr = logCorr
            bestLogOffset = offset
            bestLogData = logData

        correlationScores.append(np.linalg.norm(linearCorr))
        correlationLogScores.append(np.linalg.norm(logCorr))

    if showPlot:
        plt.plot(correlationScores)
        plt.xlabel("Cases offset (days)")
        plt.ylabel("Norm of correlation vector")
        plt.title("Linear correlation vs. data offset")
        plt.show()
        plt.plot(correlationLogScores)
        plt.xlabel("Cases offset (days)")
        plt.ylabel("Norm of correlation vector")
        plt.title("Logarithmic correlation vs. data offset")
        plt.show()

        #Plot data correlations
        #sns.pairplot(bestLinearData[['newAndPnew','retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline', 'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline','originalCases']], diag_kind='kde')
        #plt.show()

        #sns.pairplot(bestLogData[['newAndPnew','retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline', 'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline','originalCases']], diag_kind='kde')
        #plt.show()

    print("Best Full Correlation:", bestLinearCorr)
    print("Best Full Correlation Norm:", np.linalg.norm(bestLinearCorr))
    print("Best Full Offset:", bestLinearOffset)

    print("Best Log Correlation:", bestLogCorr)
    print("Best Log Correlation Norm:", np.linalg.norm(bestLogCorr))
    print("Best Log Offset:", bestLogOffset)

    #=========================BEGIN MODEL FITTING========================================

    linearMSE = []
    logMSEAdj = []
    linearCasesMSE = []
    logCasesMSE = []

    #Convert data to numpy
    linearCasesOnly = bestLinearData['originalCases'].to_numpy()
    logCasesOnly = np.log(linearCasesOnly+1)
    bestLinearData = bestLinearData.to_numpy()
    bestLogData = bestLogData.to_numpy()

    stride = 1 #trains a new model every {stride} days
    maxEpoch = 100

    for t in range((min(bestLinearData.shape[0], bestLogData.shape[0])-60)//stride):
        print("Training model:",t)

        #Linear Mobility Data
        linearTrainX = bestLinearData[t*stride:t*stride+30,1:]
        linearTrainy = bestLinearData[t*stride:t*stride+30,:1]
        linearTestX = bestLinearData[t*stride+30:t*stride+60,1:]
        linearTesty = bestLinearData[t*stride+30:t*stride+60,:1]

        #Logarithmic Mobility Data
        logTrainX = bestLogData[t*stride:t*stride+30,1:]
        logTrainy = bestLogData[t*stride:t*stride+30,:1]
        logTestX = bestLogData[t*stride+30:t*stride+60,1:]
        logTesty = bestLogData[t*stride+30:t*stride+60,:1]

        #Cases-only data
        linearCasesTrainX = linearCasesOnly[t*stride:t*stride+30]
        logCasesTrainX = logCasesOnly[t*stride:t*stride+30]
        linearCasesTestX = linearCasesOnly[t*stride+30:t*stride+60]
        logCasesTestX = logCasesOnly[t*stride+30:t*stride+60]

        timeTrain = np.arange(1,31).reshape(-1, 1)
        timeTest = np.arange(31,61).reshape(-1, 1)

        #Uncomment to add time data to mobility dataset
        #linearTrainX = np.hstack((linearTrainX, timeTrain))
        #logTrainX = np.hstack((logTrainX, timeTrain))
        #linearTestX = np.hstack((linearTestX, timeTest))
        #logTestX = np.hstack((logTestX, timeTest))

        #fit linear model
        linear_model = RidgeCV(cv=3).fit(linearTrainX, linearTrainy)

        predict = linear_model.predict(linearTestX)
        linearMSE.append(np.abs(predict-linearTesty)/linearTesty)

        #fit log model
        linear_model = RidgeCV(cv=3).fit(logTrainX, logTrainy)

        predict = linear_model.predict(logTestX)
        predictAdj = np.exp(predict)-1+np.min(full_dataframe_noDate.to_numpy()) #convert from log back to raw case number
        logMSEAdj.append(np.abs(predictAdj-linearTesty)/linearTesty)

        #fit linear cases only model
        cases_model = RidgeCV(cv=3).fit(timeTrain, linearCasesTrainX)
        if showPlot:
          visualize_cases(cases_model, timeTrain, linearCasesTrainX, timeTest, linearCasesTestX)

        predict = cases_model.predict(timeTest)
        linearCasesMSE.append(np.abs(predict-linearCasesTestX)/linearCasesTestX)

        #fit log cases only model
        cases_model = RidgeCV(cv=3).fit(np.log(timeTrain), logCasesTrainX)
        if showPlot:
          visualize_cases(cases_model, np.log(timeTrain), logCasesTrainX, np.log(timeTest), logCasesTestX)

        predict = cases_model.predict(np.log(timeTest))
        predictAdj = np.exp(predict)-1 #convert from log back to raw case number
        logCasesMSE.append(np.abs(predictAdj-linearCasesTestX)/linearCasesTestX)

    plt.plot(np.array(linearMSE).mean(axis=0), label='Mobility (linear, non-temporal)')
    plt.plot(np.array(logMSEAdj).mean(axis=0), label='Mobility (logarithmic, non-temporal)')
    plt.plot(np.array(linearCasesMSE).mean(axis=0), label='Cases (linear, temporal)')
    plt.plot(np.array(logCasesMSE).mean(axis=0), label='Cases (logarithmic temporal)')
    plt.xlabel("Days in advance to predict")
    plt.ylabel("Percent deviation from true value")
    plt.legend(loc="upper left")
    plt.show()

def visualize_cases(model, trainX, trainy, testX, testy):
  plt.scatter(trainX, trainy)
  plt.scatter(testX, testy)
  x=np.linspace(min(np.min(trainX),np.min(testX)),max(np.max(trainX),np.max(testX)),100).reshape(-1, 1)
  plt.scatter(x,model.predict(x))
  plt.show()

if __name__ == '__main__':
    baseline()