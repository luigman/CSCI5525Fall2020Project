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

    full_df=pd.read_csv('../data/COVID-19_Combined_Mobility_And_Infection_Data_Moving_Avg_updated_lin_int.csv', infer_datetime_format=True, parse_dates=True)

    #=========================FIND BEST OFFSET========================================

    by_state=full_df['sub_region_1'].unique()
    bestLinearCorr = 0
    bestLogCorr = 0
    bestLinearOffset = -1
    bestLogOffset = -1
    bestLinearData = 0
    bestLogData = 0
    #min_all_states_lin_dim=100
    #min_all_states_log_dim=100

    correlationScores = []
    correlationLogScores = []

    for offset in range(30):
        #shift all states data by offset and concatenate in order to prevent bleeding into other states' numbers
        full_dataframe=pd.DataFrame()
        min_dim=100
        for region in by_state:
            temp=full_df.loc[(full_df['sub_region_1']==region)]
            temp=temp.loc[(temp['date']>'2020-05-01')]
            #Shift CDC data by offset value
            cdc_dataframe=temp['num_cases'].shift(periods=offset,fill_value=0)
            mobility_dataframe=temp.drop(columns=['date','sub_region_1', 'num_cases'])
            all_states=pd.concat([cdc_dataframe, mobility_dataframe],axis=1)
            all_states=all_states.loc[(all_states['num_cases']>0)] #remove rows with zero cases
            full_dataframe=full_dataframe.append(all_states)
            '''if(all_states.shape[0]<min_dim):
                min_dim=all_states.shape[0]'''

        #Compute linear and logatrithmic correlations
        linearCorr = full_dataframe.corr()
        linearCorr = linearCorr.to_numpy()[0,1:] #Take only correlations between 'cases' and mobility data

        logData = np.log(full_dataframe+1-np.min(full_dataframe.to_numpy()))
        logCorr = logData.corr()
        logCorr = logCorr.to_numpy()[0,1:] #Take only correlations between 'cases' and mobility data

        #print("Offset:", offset, "Min_state_dim:    ", min_dim)
        #print("           Log Correlation:", logCorr)

        #Save best values
        if np.linalg.norm(linearCorr) > np.linalg.norm(bestLinearCorr):
            bestLinearCorr = linearCorr
            bestLinearOffset = offset
            min_all_states_lin_dim=min_dim
            #bestLinearData = full_dataframe

        if np.linalg.norm(logCorr) > np.linalg.norm(bestLogCorr):
            bestLogCorr = logCorr
            bestLogOffset = offset
            min_all_states_log_dim=min_dim
            #bestLogData = logData

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

    #num_models=(min(min_all_states_lin_dim, min_all_states_log_dim)-111)//3

    linearMSE_by_state = []
    logMSEAdj_by_state = []
    linearCasesMSE_by_state = []
    logCasesMSE_by_state = []
    logisticMSE_by_state = []
    dataNoise_by_state = []
    arimaMSE_by_state = []
    gaussMSE_by_state = []
    for s in range(len(by_state)):

        #=========================BEGIN MODEL FITTING========================================

        #Get the data for that state and shift it
        bestLinearData=pd.DataFrame()
        bestLogDf=pd.DataFrame()
        temp=full_df.loc[(full_df['sub_region_1']==by_state[s])]
        temp=temp.loc[(temp['date']<'2020-11-30')]
        #Shift CDC data by offset value
        cdc_lin_dataframe=temp['num_cases'].shift(periods=bestLinearOffset,fill_value=0)
        mobility_lin_dataframe=temp.drop(columns=['date','sub_region_1', 'num_cases'])
        all_lin_states=pd.concat([cdc_lin_dataframe, mobility_lin_dataframe],axis=1)
        all_lin_states=all_lin_states.loc[(all_lin_states['num_cases']>0)] #remove rows with zero cases
        bestLinearData=bestLinearData.append(all_lin_states)
        #Shift CDC data by offset value
        cdc_log_dataframe=temp['num_cases'].shift(periods=bestLogOffset,fill_value=0)
        mobility_log_dataframe=temp.drop(columns=['date','sub_region_1', 'num_cases'])
        all_log_states=pd.concat([cdc_log_dataframe, mobility_log_dataframe],axis=1)
        all_log_states=all_log_states.loc[(all_log_states['num_cases']>0)] #remove rows with zero cases
        bestLogDf=bestLogDf.append(all_log_states)
        bestLogData=np.log(bestLogDf+1-np.min(bestLogDf.to_numpy()))

        linearMSE = []
        logMSEAdj = []
        linearCasesMSE = []
        logCasesMSE = []
        logisticMSE = []
        dataNoise = []
        arimaMSE = []
        gaussMSE = []

        #Convert data to numpy
        linearCasesOnly = bestLinearData['num_cases'].to_numpy()
        logCasesOnly = np.log(linearCasesOnly+1)
        bestLinearData = bestLinearData.to_numpy()
        bestLogData = bestLogData.to_numpy()

        stride = 3 #trains a new model every {stride} days
        maxEpoch = 100

        for t in range((min(bestLinearData.shape[0], bestLogData.shape[0])-111)//stride):
            #print("Size of training:", range((min(bestLinearData.shape[0], bestLogData.shape[0])-111)//stride))
            print("Training model:",t)
            print("State:", by_state[s])

            #Linear Mobility Data
            linearTrainX = bestLinearData[t*stride:t*stride+60,1:]
            linearTrainy = bestLinearData[t*stride:t*stride+60,:1]
            linearTestX = bestLinearData[t*stride+60:t*stride+111,1:]
            linearTesty = bestLinearData[t*stride+60:t*stride+111,:1]

            #Logarithmic Mobility Data
            logTrainX = bestLogData[t*stride:t*stride+60,1:]
            logTrainy = bestLogData[t*stride:t*stride+60,:1]
            logTestX = bestLogData[t*stride+60:t*stride+111,1:]
            logTesty = bestLogData[t*stride+60:t*stride+111,:1]

            #Cases-only data
            linearCasesTrainX = linearCasesOnly[t*stride:t*stride+60]
            logCasesTrainX = logCasesOnly[t*stride:t*stride+60]
            linearCasesTestX = linearCasesOnly[t*stride+60:t*stride+111]
            logCasesTestX = logCasesOnly[t*stride+60:t*stride+111]

            timeTrain = np.arange(1,61).reshape(-1, 1)
            timeTest = np.arange(61,112).reshape(-1, 1)

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
            predictAdj = np.exp(predict)-1+np.min(full_dataframe.to_numpy()) #convert from log back to raw case number
            logMSEAdj.append(np.abs(predictAdj-linearTesty)/linearTesty)
        
            #fit linear cases only model
            cases_model = RidgeCV(cv=3).fit(timeTrain, linearCasesTrainX)
            if False:
                visualize_cases(cases_model, timeTrain, linearCasesTrainX, timeTest, linearCasesTestX)

            predict = cases_model.predict(timeTest)
            linearCasesMSE.append(np.abs(predict-linearCasesTestX)/linearCasesTestX)

            #fit log cases only model
            cases_model = RidgeCV(cv=3).fit(np.log(timeTrain), logCasesTrainX)
            if False:
                visualize_cases(cases_model, np.log(timeTrain), logCasesTrainX, np.log(timeTest), logCasesTestX)

            predict = cases_model.predict(np.log(timeTest))
            predictAdj = np.exp(predict)-1 #convert from log back to raw case number
            logCasesMSE.append(np.abs(predictAdj-linearCasesTestX)/linearCasesTestX)

            #fit logistic model
            logistic_model, cov = optimize.curve_fit(logisticDerivative, timeTrain.reshape(linearCasesTrainX.shape), linearCasesTrainX, p0=[4*np.max(linearCasesTrainX),60,1/30], maxfev=10000, bounds=(np.array([1, 0, 0]), np.array([20000,np.Inf,np.Inf])) )
            if False:
                visualize_logistic(logistic_model, timeTrain, linearCasesTrainX, timeTest, linearCasesTestX)

            predictLogistic = logisticDerivative(timeTest.reshape(linearCasesTestX.shape), logistic_model[0], logistic_model[1], logistic_model[2])
            logisticMSE.append(np.abs(predictLogistic-linearCasesTestX)/linearCasesTestX)

            predict = logisticDerivative(timeTrain.reshape(linearCasesTrainX.shape), logistic_model[0], logistic_model[1], logistic_model[2])
            dataNoise.append(np.mean(np.abs(predict-linearCasesTrainX)/linearCasesTrainX))

            #fit stacking regressor
            estimators = [('lr', RidgeCV()),('svr', LinearSVR(random_state=42), ('rf', RandomForestClassifier(n_estimators=10,random_state=42)))]
            reg = StackingRegressor(estimators=estimators,final_estimator=GaussianProcessRegressor(kernel=DotProduct()+WhiteKernel(),random_state=0))
            stacking_model = reg.fit(timeTrain, linearCasesTrainX)
            if False:
                visualize_cases(stacking_model, timeTrain, linearCasesTrainX, timeTest, linearCasesTestX)

            predict = stacking_model.predict(timeTest)
            linearCasesMSE.append(np.abs(predict-linearCasesTestX)/linearCasesTestX)

            #fit ARIMA
            #Perform grid search to determine ARIMA Order
            '''stepwise_fit = auto_arima(linearCasesTrainX, start_p = 1, start_q = 1, 
                            max_p = 3, max_q = 3, m = 7, 
                            start_P = 0, seasonal = True, 
                            d = None, D = 1, trace = True, 
                            error_action ='ignore',   # we don't want to know if an order does not work 
                            suppress_warnings = True,  # we don't want convergence warnings 
                            stepwise = True)           # set to stepwise 
            stepwise_fit.summary()'''

            model = SARIMAX(linearCasesTrainX, 
                    initialization='approximate_diffuse',
                    order = (2, 0, 0),  
                    seasonal_order =(2, 1, 0, 7)) 
  
            result = model.fit(disp=False) 
            if True:
                visualize_ARIMA(result, timeTrain, linearCasesTrainX, timeTest, linearCasesTestX)

            predictArima = result.predict(61, 111, typ = 'levels')
            arimaMSE.append(np.abs(predictArima-linearCasesTestX)/linearCasesTestX)

        
            #Evaluate other models to use as input to gaussian process
            arima1 = SARIMAX(linearCasesTrainX, initialization='approximate_diffuse', order = (2, 0, 0), seasonal_order =(2, 1, 0, 7)).fit(disp=False)
            arima2 = SARIMAX(linearCasesTrainX, initialization='approximate_diffuse', order = (2, 0, 0), seasonal_order =(2, 1, 1, 7)).fit(disp=False)
            arima3 = SARIMAX(linearCasesTrainX, initialization='approximate_diffuse', order = (1, 1, 0), seasonal_order =(1, 1, 1, 7)).fit(disp=False)
            arima4 = SARIMAX(linearCasesTrainX, initialization='approximate_diffuse', order = (0, 1, 1), seasonal_order =(1, 1, 1, 7)).fit(disp=False)
            arima5 = SARIMAX(linearCasesTrainX, initialization='approximate_diffuse', order = (0, 1, 1), seasonal_order =(2, 1, 0, 7)).fit(disp=False)

            predictLog = cases_model.predict(np.log(timeTrain)) #Log model
            predictAdj = np.exp(predictLog)-1 #convert from log back to raw case number
            predictLogistic = logisticDerivative(timeTrain.reshape(linearCasesTrainX.shape), logistic_model[0], logistic_model[1], logistic_model[2]) #logistic model
            predictArima1 = arima1.predict(1, 60, typ = 'levels')
            predictArima2 = arima2.predict(1, 60, typ = 'levels')
            predictArima3 = arima3.predict(1, 60, typ = 'levels')
            predictArima4 = arima4.predict(1, 60, typ = 'levels')
            predictArima5 = arima5.predict(1, 60, typ = 'levels')

            testLog = cases_model.predict(np.log(timeTest)) #Log model
            testAdj = np.exp(testLog)-1 #convert from log back to raw case number
            testLogistic = logisticDerivative(timeTest.reshape(linearCasesTestX.shape), logistic_model[0], logistic_model[1], logistic_model[2]) #logistic model
            testArima1 = arima1.predict(61, 111, typ = 'levels')
            testArima2 = arima2.predict(61, 111, typ = 'levels')
            testArima3 = arima3.predict(61, 111, typ = 'levels')
            testArima4 = arima4.predict(61, 111, typ = 'levels')
            testArima5 = arima5.predict(61, 111, typ = 'levels')

            #fit gaussian process meta-learner
            gaussTrain = np.array([predictLogistic, predictArima1, predictArima2, predictArima3, predictArima4, predictArima5]).T
            gaussTest = np.array([testLogistic, testArima1, testArima2, testArima3, testArima4, testArima5]).T
            reg = GaussianProcessRegressor(kernel=DotProduct()+WhiteKernel(),random_state=0)
            stacking_model = reg.fit(gaussTrain, linearCasesTrainX)
            predictTrain = stacking_model.predict(gaussTrain)
            predictTest = stacking_model.predict(gaussTest)
            if False:
                visualize_gauss(np.hstack((predictTrain, predictTest)).T, timeTrain, linearCasesTrainX, timeTest, linearCasesTestX)

            gaussMSE.append(np.abs(predictTest-linearCasesTestX)/linearCasesTestX)

        #Append to state totals
        linearMSE_by_state.append(np.reshape(np.array(linearMSE).mean(axis=0), (51)))
        logMSEAdj_by_state.append(np.reshape(np.array(logMSEAdj).mean(axis=0), (51)))
        linearCasesMSE_by_state.append(np.reshape(np.array(linearCasesMSE).mean(axis=0), (51)))
        logCasesMSE_by_state.append(np.reshape(np.array(logCasesMSE).mean(axis=0), (51)))
        logisticMSE_by_state.append(np.reshape(np.array(logisticMSE).mean(axis=0), (51)))
        dataNoise_by_state.append(np.mean(dataNoise))
        arimaMSE_by_state.append(np.reshape(np.array(arimaMSE).mean(axis=0), (51)))
        gaussMSE_by_state.append(np.reshape(np.array(gaussMSE).mean(axis=0), (51)))
        print("Average logistic Test error:", np.mean(dataNoise))

    #Plot proof-of-concept graph
    if showPlot:
        plt.plot(np.array(linearMSE_by_state).mean(axis=0), label='Mobility (linear, non-temporal)')
        plt.plot(np.array(logMSEAdj_by_state).mean(axis=0), label='Mobility (logarithmic, non-temporal)')
        plt.xlabel("Days in advance to predict")
        plt.ylabel("Percent deviation from true value")
        plt.legend(loc="upper left")
        plt.show()
      
        #Plot baseline graph
        #plt.plot(np.array(linearCasesMSE_by_state).mean(axis=0), label='Cases (linear, temporal)') #Don't plot because performance is terrible
        plt.plot(np.array(logCasesMSE_by_state).mean(axis=0), label='Cases (logarithmic temporal)')
        plt.plot(np.array(logisticMSE_by_state).mean(axis=0), label='Cases (logistic temporal)')
        plt.plot(np.array(arimaMSE_by_state).mean(axis=0), label='Cases (ARIMA)')
        plt.plot(np.array(gaussMSE_by_state).mean(axis=0), label='Cases (Gaussian Process meta)')
        plt.xlabel("Days in advance to predict")
        plt.ylabel("Percent deviation from true value")
        plt.legend(loc="upper left")
        plt.show()
    print("Average logistic test error:", np.mean(dataNoise_by_state))


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
    full_predictions = model.predict(1, 112, typ = 'levels')

    plt.plot(full_predictions)
    plt.scatter(trainX,trainy)
    plt.scatter(testX,testy)
    plt.title("ARIMA Projection of COVID case numbers")
    plt.xlabel("Time (days)")
    plt.ylabel("Number of cases per 100,000 people")
    plt.show()

def visualize_gauss(predictions, trainX, trainy, testX, testy):
  plt.scatter(trainX, trainy)
  plt.scatter(testX, testy)
  x=np.linspace(min(np.min(trainX),np.min(testX)),max(np.max(trainX),np.max(testX)),100).reshape(-1, 1)
  plt.scatter(np.vstack((trainX, testX)),predictions)
  plt.show()

if __name__ == '__main__':
    baseline(True)