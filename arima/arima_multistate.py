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

    #Find length of shorted state dataframe
    minLength = np.inf
    for region in by_state:
        state_data=full_dataframe.loc[(full_dataframe['sub_region_1']==region)]
        length = state_data.shape[0]
        if length < minLength:
            minLength = length

    stride = 10 #trains a new model every {stride} days
    percentErrors = []
    for t in range(3):#(minLength-90)//stride):
        #Linear Mobility Data
        linearTrainX = []
        linearTrainy = []
        linearTestX = []
        linearTesty = []

        #Logarithmic Mobility Data
        logTrainX = []
        logTrainy = []
        logTestX = []
        logTesty = []

        MLPTrainX = []

        for region in by_state[:3]:
            state_data=full_dataframe.loc[(full_dataframe['sub_region_1']==region)].drop(columns=['sub_region_1', 'grocery_and_pharmacy_percent_change_from_baseline'])
            #Convert data to numpy
            linearData = state_data.to_numpy()
            logData = np.log(state_data+1-np.min(state_data.to_numpy())).to_numpy()

            timeTrain = np.arange(1,61).reshape(-1, 1)
            timeTest = np.arange(61,91).reshape(-1, 1)
        
            #Linear Mobility Data
            linearTrainX.append(linearData[t*stride:t*stride+60,1:])
            linearTrainy.append(linearData[t*stride:t*stride+60,:1])
            linearTestX.append(linearData[t*stride+60:t*stride+90,1:])
            linearTesty.append(linearData[t*stride+60:t*stride+90,:1])

            #Logarithmic Mobility Data
            logTrainX.append(logData[t*stride:t*stride+60,1:])
            logTrainy.append(logData[t*stride:t*stride+60,:1])
            logTestX.append(logData[t*stride+60:t*stride+90,1:])
            logTesty.append(logData[t*stride+60:t*stride+90,:1])

            
            MLPTrainXState = []
            for i,feature in enumerate(linearData[t*stride:t*stride+60,1:].T):
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
                arimaOrders = [(1,0,0),(1,0,1),(3,0,0),(1,0,0),(0,1,1),(1,0,0),(2,0,0)]
                seasonalOrders = [(2, 1, 0, 7), (2, 1, 0, 7), (1, 1, 0, 7), (1, 1, 0, 7),(0,1,1,7),(0,1,1,7),(2, 1, 0, 7)]

                model = SARIMAX(feature,  
                        order = arimaOrders[i],  
                        seasonal_order =seasonalOrders[i],
                        initialization='approximate_diffuse') 
        
                result = model.fit(disp=False) 
                if showPlot >=2 :
                    visualize_ARIMA(result, timeTrain, linearTrainX[:,i], timeTest, linearTestX[:,i])

                predictArima.append(result.predict(61, 90, typ = 'levels'))
                predictArima = np.mean(predictArima, axis=0)
                MLPTrainXState.append(predictArima)
            MLPTrainX.append(np.array(MLPTrainXState).T)
        MLPTrainX = np.array(MLPTrainX).reshape(-1,6)
        linearTrainX = np.array(linearTrainX).reshape(-1,6)
        linearTrainy = np.array(linearTrainy).reshape(-1,1)
        linearTesty = np.array(linearTesty).reshape(-1,1)

        #Use "Last known case value" as bias
        #(I completely made this up but it improved accuracy by ~5%)
        #bias1 = np.ones((30,1))#*linearTrainy[0]
        #bias2 = np.ones((30,1))#*linearTrainy[30]
        bias = np.ones((linearTrainX.shape[0],1))#np.vstack((bias1, bias2))
        linearTrainX = np.hstack((linearTrainX, bias))

        bias3 = np.ones((MLPTrainX.shape[0],1))#*linearTrainy[-1]
        MLPTrainX = np.hstack((MLPTrainX, bias3))
        
        failCounter = 0
        maxFail = 4
        while failCounter < maxFail: #Retrain if prediction is zero
            model = Sequential()
            #model.add(BatchNormalization())
            model.add(Dense(10, input_dim=7, activation='relu'))
            #model.add(Dropout(0.15))
            model.add(Dense(30, activation='relu'))
            #model.add(Dropout(0.15))
            model.add(Dense(1, activation='relu'))

            model.compile(optimizer='adam',loss='mean_squared_error', metrics=['accuracy'])
            model.fit(linearTrainX, linearTrainy, epochs=100, verbose=0)

            y_pred = model.predict(MLPTrainX)
            if np.sum(y_pred==0) < 0.1 * MLPTrainX.shape[0]:
                break
            print("Prediction is zero. Retraining...")
            failCounter += 1
            if failCounter == maxFail:
                failedMonths += 1
                percentError = 1
                print("Could not train model on this data")
        if failCounter != maxFail:
            error = y_pred-linearTesty
            percentError = np.abs(error/linearTesty).T
            percentErrorsByState = []
            print(percentError.shape)
            for i in range(len(by_state)):
                percentErrorsByState.append(percentError[i*30:(i+1)*30])
            percentErrorsByState = np.array(percentErrorsByState).reshape(51)
            print("Loss:", np.mean(percentError))
            #print("Percent Error:",percentError)
            percentErrors.append(percentErrorsByState)

        if showPlot >= 1 or np.mean(percentError) > 0.4:
            plt.plot(timeTrain, linearTrainy[0:60], label="Past")
            plt.plot(timeTest, linearTesty[0:30], label="True Future")
            plt.plot(timeTest, y_pred[0:30], label="Predicted Future")
            plt.plot(timeTest, MLPTrainX[0:30,-2], label="Predicted ARIMA (case only)")
            plt.legend()
            plt.show()
    print(np.array(percentErrors).shape)
    print("Failed Months:", failedMonths)
    print(np.mean(percentErrors, axis=1))
    plt.plot(np.mean(percentErrors, axis=1).flatten())
    plt.show()
    return

def visualize_ARIMA(model, trainX, trainy, testX, testy):
    full_predictions = model.predict(1, 91, typ = 'levels')

    plt.plot(full_predictions)
    plt.scatter(trainX,trainy)
    plt.scatter(testX,testy)
    plt.show()

arima()