import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

def baseline():
    showPlot = True
    np.set_printoptions(precision=3, suppress=True)

    mobility_dataframe = pd.read_csv('google_baseline_test.csv', infer_datetime_format=True, parse_dates=True)
    cdc_dataframe = pd.read_csv('cdc_baseline_test_movingAvg.csv', infer_datetime_format=True, parse_dates=True)

    full_dataframe = pd.concat([mobility_dataframe, cdc_dataframe], axis=1)

    #sns.pairplot(full_dataframe[['newAndPnew', 'retail_and_recreation_percent_change_from_baseline', 'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline']], diag_kind='kde')
    #plt.show()

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
        full_dataframe_noDate = full_dataframe.drop(columns=['submission_date'])
        full_dataframe_noDate = full_dataframe_noDate.loc[(full_dataframe_noDate['newAndPnew']!=0)] #remove rows with zero cases

        #Compute linear and logatrithmic correlations
        linearCorr = full_dataframe_noDate.corr()
        linearCorr = linearCorr.to_numpy()[0,0:] #Take only correlations between 'cases' and mobility data

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

        sns.pairplot(bestLinearData[['newAndPnew','retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline', 'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline']], diag_kind='kde')
        plt.show()

        sns.pairplot(bestLogData[['newAndPnew','retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline', 'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline']], diag_kind='kde')
        plt.show()

    print("Best Full Correlation:", linearCorr)
    print("Best Full Correlation Norm:", bestLinearCorr)
    print("Best Full Offset:", bestLinearOffset)

    print("Best Log Correlation:", bestLogCorr)
    print("Best Log Correlation Norm:", np.linalg.norm(bestLogCorr))
    print("Best Log Offset:", bestLogOffset)

    #Define models

    normalizer = preprocessing.Normalization()

    linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
    ])

    linear_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')

    dnn_model = keras.Sequential([
      normalizer,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
    ])

    dnn_model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))

    linearMSE = []
    logMSE = []
    logMSEAdj = []
    linearDNNMSE = []
    logDNNMSE = []
    logDNNMSEAdj = []

    #Convert data to numpy
    bestLinearData = bestLinearData.to_numpy()
    bestLogData = bestLogData.to_numpy()

    stride = 30 #trains a new model every {stride} days
    print((min(bestLinearData.shape[0], bestLogData.shape[0])-30)//stride)

    for t in range((min(bestLinearData.shape[0], bestLogData.shape[0])-30)//stride):
        linearTrainX = bestLinearData[t*stride:(t+1)*stride,1:]
        linearTrainy = bestLinearData[t*stride:(t+1)*stride,:1]
        logTrainX = bestLogData[t*stride:(t+1)*stride,1:]
        logTrainy = bestLogData[t*stride:(t+1)*stride,:1]

        linearTestX = bestLinearData[(t+1)*stride:(t+2)*stride,1:]
        linearTesty = bestLinearData[(t+1)*stride:(t+2)*stride,:1]
        logTestX = bestLogData[(t+1)*stride:(t+2)*stride,1:]
        logTesty = bestLogData[(t+1)*stride:(t+2)*stride,:1]

        #fit linear model
        linHistory = linear_model.fit(linearTrainX, linearTrainy, epochs=100,verbose=0)

        evaluate = linear_model.evaluate(linearTestX, linearTesty, verbose=0)
        predict = linear_model.predict(linearTestX, verbose=0)
        linearMSE.append(np.abs(predict-linearTesty))

        reset_weights(linear_model)

        #fit log model
        logHistory = linear_model.fit(logTrainX, logTrainy, epochs=100,verbose=0)

        evaluate = linear_model.evaluate(logTestX, logTesty, verbose=0)
        predict = linear_model.predict(logTestX, verbose=0)
        
        predictAdj = np.exp(predict)-1+np.min(full_dataframe_noDate.to_numpy())
        #print(np.exp(prediction)-1+np.min(full_dataframe_noDate))
        logMSE.append(np.abs(predict-logTesty))
        logMSEAdj.append(np.abs(predictAdj-linearTrainy))

        reset_weights(linear_model)

        #fit linear DNN model
        linHistory = dnn_model.fit(linearTrainX, linearTrainy, epochs=100,verbose=0)

        evaluate = dnn_model.evaluate(linearTestX, linearTesty, verbose=0)
        predict = dnn_model.predict(linearTestX, verbose=0)
        linearDNNMSE.append(np.abs(predict-linearTesty))

        reset_weights(dnn_model)

        #fit log DNN model
        logHistory = dnn_model.fit(logTrainX, logTrainy, epochs=100,verbose=0)

        evaluate = dnn_model.evaluate(logTestX, logTesty, verbose=0)
        predict = dnn_model.predict(logTestX, verbose=0)
        
        predictAdj = np.exp(predict)-1+np.min(full_dataframe_noDate.to_numpy())
        #print(np.exp(prediction)-1+np.min(full_dataframe_noDate))
        logDNNMSE.append(np.abs(predict-logTesty))
        logDNNMSEAdj.append(np.abs(predictAdj-linearTrainy))

        reset_weights(dnn_model)
        
        
    plt.plot(np.array(linearMSE).mean(axis=0), label='Linear')
    plt.plot(np.array(logMSEAdj).mean(axis=0), label='Log Adjusted')
    plt.plot(np.array(linearDNNMSE).mean(axis=0), label='Linear DNN')
    plt.plot(np.array(logDNNMSEAdj).mean(axis=0), label='Log DNN Adjusted')
    plt.legend(loc="upper left")
    plt.show()


#Reset weights from:
#https://github.com/keras-team/keras/issues/341
def reset_weights(model):
  for layer in model.layers: 
    if isinstance(layer, tf.keras.Model):
      reset_weights(layer)
      continue
    for k, initializer in layer.__dict__.items():
      if "initializer" not in k:
        continue
      # find the corresponding variable
      var = getattr(layer, k.replace("_initializer", ""))
      var.assign(initializer(var.shape, var.dtype))


if __name__ == '__main__':
    baseline()