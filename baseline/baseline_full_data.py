import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import StackingRegressor, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from pmdarima import auto_arima
# Ignore harmless warnings 
import warnings 
warnings.filterwarnings("ignore") 
from statsmodels.tsa.statespace.sarimax import SARIMAX 

from scipy import optimize

def baseline(showPlot):
    np.set_printoptions(precision=3, suppress=True)

    full_df=pd.read_csv('../data/COVID-19_Combined_Mobility_And_Infection_Data_Moving_Avg.csv', infer_datetime_format=True, parse_dates=True)

    #=========================FIND BEST OFFSET========================================

    by_state=full_df['sub_region_1'].unique()
    linear_scores_by_state={}
    log_scores_by_state={}

    for region in by_state:

        temp=full_df.loc[(full_df['sub_region_1']==region)]

        bestLinearCorr = 0
        bestLogCorr = 0
        bestLinearOffset = -1
        bestLogOffset = -1
        bestLinearData = 0
        bestLogData = 0

        correlationScores = []
        correlationLogScores = []

        for offset in range(100):
            #Shift CDC data by offset value - this is going to create some problems because we'll have to do if for each state...
            cdc_dataframe=temp['num_cases'].shift(periods=offset,fill_value=0)

            #Build new full data array
            mobility_dataframe=temp.drop(columns=['date','sub_region_1', 'num_cases'])
            full_dataframe=pd.concat([cdc_dataframe, mobility_dataframe],axis=1)
            full_dataframe['originalCases'] = temp['num_cases'] #preserve original case values as additional feature
            full_dataframe=full_dataframe.loc[(full_dataframe['num_cases']!=0)] #remove rows with zero cases

            #Compute linear and logatrithmic correlations
            linearCorr = full_dataframe.corr()
            linearCorr = linearCorr.to_numpy()[0,1:] #Take only correlations between 'cases' and mobility data

            logData = np.log(full_dataframe+1-np.min(full_dataframe.to_numpy()))
            logCorr = logData.corr()
            logCorr = logCorr.to_numpy()[0,1:] #Take only correlations between 'cases' and mobility data

            print("Offset:", offset, "Correlation:    ", linearCorr)
            print("           Log Correlation:", logCorr)

            #Save best values
            if np.linalg.norm(linearCorr) > np.linalg.norm(bestLinearCorr):
                bestLinearCorr = linearCorr
                bestLinearOffset = offset
                bestLinearData = full_dataframe

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

        print("Best Full Correlation:", bestLinearCorr)
        print("Best Full Correlation Norm:", np.linalg.norm(bestLinearCorr))
        print("Best Full Offset:", bestLinearOffset)

        print("Best Log Correlation:", bestLogCorr)
        print("Best Log Correlation Norm:", np.linalg.norm(bestLogCorr))
        print("Best Log Offset:", bestLogOffset)

        linear_scores_by_state[region]=bestLinearOffset
        log_scores_by_state[region]=bestLogOffset
    print(linear_scores_by_state)
    print(log_scores_by_state)





def logisticDerivative(x, a, b, c):
    y = a*np.exp(c*(b-x))/(1+np.exp(c*(b-x)))**2
    return y

def visualize_cases(model, trainX, trainy, testX, testy):
  plt.scatter(trainX, trainy)
  plt.scatter(testX, testy)
  x=np.linspace(min(np.min(trainX),np.min(testX)),max(np.max(trainX),np.max(testX)),100).reshape(-1, 1)
  plt.scatter(x,model.predict(x))
  plt.show()

def visualize_logistic(model, trainX, trainy, testX, testy):
  plt.scatter(trainX, trainy)
  plt.scatter(testX, testy)
  x=np.linspace(min(np.min(trainX),np.min(testX)),max(np.max(trainX),np.max(testX)),100).reshape(-1, 1)
  plt.scatter(x,logisticDerivative(x, model[0], model[1], model[2]))
  plt.show()

def visualize_ARIMA(model, trainX, trainy, testX, testy):
    full_predictions = model.predict(1, 91, typ = 'levels')

    plt.plot(full_predictions)
    plt.scatter(trainX,trainy)
    plt.scatter(testX,testy)
    plt.show()

def visualize_gauss(predictions, trainX, trainy, testX, testy):
  plt.scatter(trainX, trainy)
  plt.scatter(testX, testy)
  x=np.linspace(min(np.min(trainX),np.min(testX)),max(np.max(trainX),np.max(testX)),100).reshape(-1, 1)
  plt.scatter(np.vstack((trainX, testX)),predictions)
  plt.show()

if __name__ == '__main__':
    baseline(False)