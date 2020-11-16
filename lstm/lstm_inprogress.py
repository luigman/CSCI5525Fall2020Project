import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

#USEFUL LINKS
#useful forecasting lstm info
#https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/#:~:text=The%20Long%20Short%2DTerm%20Memory,useful%20for%20time%20series%20forecasting.
#different methods for forecasting
#https://machinelearningmastery.com/multi-step-time-series-forecasting/


#make data stationary just the covid cases number or all the features?
#subtracts t-1 from t to create stationary data
def stationary(data):
    stationary_data = np.zeros(np.shape(data))
    stationary_data = np.delete(stationary_data, len(stationary_data[0])-1, 1)
    for i in range(0,len(data)):
        for j in range(1,len(data[0])):
            stationary_data[i][j-1] = np.subtract(data[i][j], data[i][j-1])
    return stationary_data

#normalize all of the features between -1 and 1
def normalize(data):
    #find total min of each feature
    #find total max of each feature
    print(np.shape(np.ndarray.min(data,(0,1))))
    print(np.shape(np.ndarray.max(data,(0,1))))
    min_vals = np.ndarray.min(data,(0,1))
    max_vals = np.ndarray.max(data,(0,1))

    #normalize between -1 and 1
    normalized_data = np.zeros(np.shape(data))
    for i in range(0,len(data)):
        for j in range(0,len(data[0])):
            normalized_data[i][j] = (((data[i][j] - min_vals) / (max_vals - min_vals)) * 2) - 1

    return normalized_data

#sample smaller sequence lengths from the longer ones to create more learnable data
def sample(data, num_samples, num_timesteps):
    samples_x = np.zeros((num_samples,num_timesteps,7))
    samples_y = np.zeros((num_samples,1,7))
    for i in range(0,num_samples):
        state = random.randint(0,50)
        day = random.randint(0,len(data[0])-(num_timesteps+1))
        samples_x[i] = data[state][day:day+num_timesteps]
        samples_y[i] = data[state][day+num_timesteps]
    return samples_x, samples_y



def lstm():
    df = pd.read_csv(r'COVID-19_Combined_Mobility_And_Infection_Data.csv')
    states = df.sub_region_1.unique()

    #Format Data for LSTM Input
    #data in the shape[51,266,7] or [states,days,features]
    df_list = [0] * 51
    data = [0] * 51
    for i in range(0,len(states)):
        df_list[i] = df.loc[df['sub_region_1'] == states[i]]
        df_list[i] = df_list[i].drop(columns=['date'])
        df_list[i] = df_list[i].drop(columns=['sub_region_1'])
        data[i] = df_list[i].to_numpy()


    #For now replace nan with 0 but in future replace with avg of before and after nan?
    print("Number of nan in data to be replaced with 0: ", np.count_nonzero(np.isnan(data)))
    data = np.nan_to_num(data)

    #Stationarize Data Here
    stationary_data = stationary(data)

    #Normalize Data Between -1 and 1 Here
    normalized_data = normalize(stationary_data)

    data = normalized_data

    ### COMMENTED OUT for now as went with different approach to creating labels for lstm-------
    # #Create labels from time sequenced data by offset number of days
    # OFFSET = 1
    # TT_SPLIT = 200
    #
    # data = np.asarray(data)
    # data_x = np.asarray(data)
    # data_y = np.zeros((51,len(data_x[0]),7))
    # for i in range(0,51):
    #     for j in range(0,len(data_x[0])-OFFSET):
    #         for k in range(0,7):
    #             data_y[i][j][k] = np.asarray(data_x[i][j+OFFSET][k])
    #
    # data_x = np.delete(data_x, slice(len(data_x[0])-OFFSET,len(data_x[0])), 1)
    # data_y = np.delete(data_y, slice(len(data_y[0])-OFFSET,len(data_y[0])), 1)
    #
    # #Split data into training and testing
    # train_x, test_x = np.split(data_x, [TT_SPLIT], 1)
    # train_y, test_y = np.split(data_y, [TT_SPLIT], 1)
    #
    # print(np.shape(train_x))
    # print(np.shape(train_y))
    # print(np.shape(test_x))
    # print(np.shape(test_y))
    ### -------------------------------------------------------------------------------------

    #Create training and labels for lstm by sampling from longer sequence of 266 Days
    #takes 1000 samples of 20 day segments and the 21st day is the label
    TT_SPLIT = 200
    train_x, test_x = np.split(data, [TT_SPLIT], 1)
    train_sample_x, train_sample_y = sample(train_x, 1000, 20)
    test_sample_x, test_sample_y = sample(test_x, 100, 20)

    #Create model
    #Haven't tuned model at all.  Need to try different structures/parameters
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(20, input_shape=(len(train_sample_x[0]), 7), return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(20, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(20, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(7))


    model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.001), metrics = ['accuracy'])

    print(model.summary())

    model.fit(train_sample_x, train_sample_y, epochs=50, batch_size = 1, validation_split = 0.1)

    #once we have a semi accurate fit model
    #to predict multiple timesteps in the future
    #USE EITHER
    #Direct Multi-step Forecast
    #OR
    #Recursive Strategy

    #Recursive works by predicting one step ahead and using that data to predict another step ahead...
    #Direct works by training directly using desired step as label

    #once values are predicted, to understand prediction
    #un-normalize data
    #un-stationarize data

if __name__ == '__main__':
    lstm()
