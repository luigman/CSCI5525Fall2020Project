import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pmdarima import auto_arima
import statsmodels.api as sm
# Ignore harmless warnings 
import warnings 
warnings.filterwarnings("ignore") 
from statsmodels.tsa.statespace.sarimax import SARIMAX 

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.utils import to_categorical

np.set_printoptions(precision=3, suppress=True)
showPlot = 0

def arima():
    failedMonths = 0 #Records if any months could not be successfully trained on (pred is zero)

    full_df=pd.read_csv('../data/COVID-19_Combined_Mobility_And_Infection_Data_Moving_Avg_updated_mode.csv', infer_datetime_format=True, parse_dates=True)
    full_df['originalCases'] = full_df['num_cases'] #preserve original case values as additional feature

    by_state=full_df['sub_region_1'].unique()
    

    #shift all states data by offset and concatenate in order to prevent bleeding into other states' numbers
    offset = 14
    full_dataframe=pd.DataFrame()
    for region in by_state:
        temp=full_df.loc[(full_df['sub_region_1']==region)]
        temp=temp.loc[(temp['date']<'2020-11-20')]
        #Shift CDC data by offset value
        cdc_dataframe=temp['num_cases'].shift(periods=offset,fill_value=0)
        mobility_dataframe=temp.drop(columns=['date', 'num_cases'])
        all_states=pd.concat([cdc_dataframe, mobility_dataframe],axis=1)
        all_states=all_states.loc[(all_states['num_cases']>0)] #remove rows with zero cases
        full_dataframe=full_dataframe.append(all_states)

    #Build new full data array
    #mobility_dataframe_truc = mobility_dataframe.drop(columns=['date'])
    #full_dataframe = pd.concat([cdc_dataframe_truc, mobility_dataframe_truc], axis=1)
    #full_dataframe['originalCases'] = cdc_dataframe['newAndPnew'] #preserve original case values as additional feature
    #full_dataframe_noDate = full_dataframe.drop(columns=['submission_date'])
    #full_dataframe_noDate = full_dataframe_noDate.loc[(full_dataframe_noDate['newAndPnew']!=0)] #remove rows with zero cases

    for region in by_state:
        region='Minnesota'
        state_data=full_dataframe.loc[(full_dataframe['sub_region_1']==region)].drop(columns=['sub_region_1'])
        #Convert data to numpy
        linearData = state_data.to_numpy()
        logData = np.log(state_data+1-np.min(state_data.to_numpy())).to_numpy()
        
        stride = 10 #trains a new model every {stride} days
        percentErrors = []
        timeTrain = np.arange(1,61).reshape(-1, 1)
        timeTest = np.arange(61,91).reshape(-1, 1)

        for t in range((min(linearData.shape[0], logData.shape[0])-90)//stride):
            #Linear Mobility Data
            linearTrainX = linearData[t*stride:t*stride+60,1:]
            linearTrainy = linearData[t*stride:t*stride+60,:1]
            linearTestX = linearData[t*stride+60:t*stride+90,1:]
            linearTesty = linearData[t*stride+60:t*stride+90,:1]

            #Logarithmic Mobility Data
            logTrainX = logData[t*stride:t*stride+60,1:]
            logTrainy = logData[t*stride:t*stride+60,:1]
            logTestX = logData[t*stride+60:t*stride+90,1:]
            logTesty = logData[t*stride+60:t*stride+90,:1]

            
            MLPTrainX = []
            for i,feature in enumerate(linearTrainX.T):
                #print("Feature:", i)
                #fit ARIMA
                #Perform grid search to determine ARIMA Order
                #stepwise_fit = auto_arima(feature, start_p = 1, start_q = 1, 
                #                max_p = 3, max_q = 3, m = 7, 
                #                start_P = 0, seasonal = True, 
                #                d = None, D = 1, trace = True, 
                #                error_action ='ignore',   # we don't want to know if an order does not work 
                #                suppress_warnings = True,  # we don't want convergence warnings 
                #                stepwise = True)           # set to stepwise 
                #stepwise_fit.summary() 
                #print("===============================================================================================")
                
                predictArima =[]
                arimaOrders = [(2,0,0), (2,1,0), (0,1,1)]
                seasonalOrders = [(2, 1, 0, 7), (0,1,1,7)]
                for order in arimaOrders:
                    for seasonalOrder in seasonalOrders:
                        model = SARIMAX(feature,  
                                order = order,  
                                seasonal_order =seasonalOrder,
                                initialization='approximate_diffuse') 
                
                        result = model.fit(disp=False) 
                        if showPlot >=2 :
                            visualize_ARIMA(result, timeTrain, linearTrainX[:,i], timeTest, linearTestX[:,i])

                        predictArima.append(result.predict(61, 120, typ = 'levels'))
                predictArima = np.mean(predictArima, axis=0)
                MLPTrainX.append(predictArima)
            MLPTrainX = np.array(MLPTrainX).T

            #Use "Last known case value" as bias
            #(I completely made this up but it improved accuracy by ~5%)
            bias1 = np.ones((30,1))*linearTrainy[0]
            bias2 = np.ones((30,1))*linearTrainy[30]
            bias = np.vstack((bias1, bias2))
            linearTrainX = np.hstack((linearTrainX, bias))

            bias3 = np.ones((60,1))*linearTrainy[-1]
            MLPTrainX = np.hstack((MLPTrainX, bias3))
            
            failCounter = 0
            maxFail = 4
            while failCounter < maxFail: #Retrain if prediction is zero
                model = Sequential()
                #model.add(BatchNormalization())
                model.add(Dense(10, input_dim=8, activation='relu'))
                #model.add(Dropout(0.15))
                model.add(Dense(30, activation='relu'))
                #model.add(Dropout(0.15))
                model.add(Dense(1, activation='relu'))

                model.compile(optimizer='adam',loss='mean_squared_error', metrics=['accuracy'])
                model.fit(linearTrainX, linearTrainy, epochs=100, verbose=0)

                y_pred = model.predict(MLPTrainX)[0:30]
                if np.sum(y_pred==0) == 0:
                    break
                print("Prediction is zero. Retraining...")
                failCounter += 1
                if failCounter == maxFail:
                    failedMonths += 1
                    print("Could not train model on this data")
            if failCounter != maxFail:
                error = y_pred-linearTesty
                percentError = np.abs(error/linearTesty).T
                print("Loss:", np.mean(percentError))
                #print("Percent Error:",percentError)
                percentErrors.append(percentError)

            if showPlot >= 1 or np.mean(percentError) > 0.4:
                plt.plot(timeTrain, linearTrainy, label="Past")
                plt.plot(timeTest, linearTesty, label="True Future")
                plt.plot(timeTest, y_pred, label="Predicted Future")
                plt.plot(timeTest, MLPTrainX[0:30,-2], label="Predicted ARIMA (case only)")
                plt.legend()
                plt.show()

        print("Failed Months:", failedMonths)
        print(np.mean(percentErrors, axis=0))
        plt.plot(np.mean(percentErrors, axis=0).flatten())
        plt.show()
    return

def visualize_ARIMA(model, trainX, trainy, testX, testy):
    full_predictions = model.predict(1, 91, typ = 'levels')

    plt.plot(full_predictions)
    plt.scatter(trainX,trainy)
    plt.scatter(testX,testy)
    plt.show()

arima()