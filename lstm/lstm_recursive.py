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

#IF load is True then it loads in a previously trained model
LOAD = True
#make data stationary just the covid cases number or all the features?
#subtracts t-1 from t to create stationary data
def stationary(data):
    stationary_data = np.zeros(np.shape(data))
    stationary_data = np.delete(stationary_data, len(stationary_data[0])-1, 1)
    start_point = np.zeros((len(data), len(data[0][0])))
    for i in range(0,len(data)):
        start_point[i] = data[i][0]
        for j in range(1,len(data[0])):
            stationary_data[i][j-1] = data[i][j]
            stationary_data[i][j-1][6] = np.subtract(data[i][j][6], data[i][j-1][6])
    return stationary_data, start_point

#normalize all of the features between -1 and 1
def normalize(data, min_vals, max_vals):
    #normalize between -1 and 1
    normalized_data = np.zeros(np.shape(data))
    for i in range(0,len(data)):
        for j in range(0,len(data[0])):
            #normalized_data[i][j] = (((data[i][j] - min_vals) / (max_vals - min_vals)) * 2) - 1
            normalized_data[i][j] = ((data[i][j] - min_vals) / (max_vals - min_vals))
    return normalized_data, min_vals, max_vals

#record initial starting point, add change to starting point for each timestep to rebuild data
def undo_stationary(data, start_point):
    #print(np.shape(data))
    stationary_data = np.zeros((np.shape(data)[0], np.shape(data)[1] + 1, np.shape(data)[2]))
    #print(np.shape(stationary_data))
    for i in range(0,len(data)):
        stationary_data[i][0] = start_point[i]
        for j in range(1,len(stationary_data[0])):
            stationary_data[i][j] = data[i][j-1]
            stationary_data[i][j][6] = np.add(stationary_data[i][j-1][6], data[i][j-1][6])
    return stationary_data

def undo_normalize(norm_data, min_vals, max_vals):
    undo_normalized_data = np.zeros(np.shape(norm_data))
    for i in range(0,len(norm_data)):
        for j in range(0,len(norm_data[0])):
            #undo_normalized_data[i][j] = (((norm_data[i][j] + 1.0) / 2.0) * (max_vals - min_vals)) + min_vals
            undo_normalized_data[i][j] = ((norm_data[i][j]) * (max_vals - min_vals)) + min_vals
    return undo_normalized_data

#sample smaller sequence lengths from the longer ones to create more learnable data
def sample(data, num_samples, num_timesteps):
    samples_x = np.zeros((num_samples,num_timesteps,7))
    samples_y = np.zeros((num_samples,1,7))
    for i in range(0,num_samples):
        state = random.randint(0,len(data)-1)
        day = random.randint(0,len(data[0])-(num_timesteps+1))
        samples_x[i] = data[state][day:day+num_timesteps]
        samples_y[i] = data[state][day+num_timesteps]
    return samples_x, samples_y

#predict a certain number of steps in the future
# prev - previous time steps to predict on
# num_steps_pred - how many steps in the future to predict
def predict(prev, num_steps_pred, model):
    prev_new = np.reshape(prev, (1,len(prev),7))
    prediction = np.zeros((num_steps_pred,7))
    prediction[0] = model.predict(prev_new)
    for j in range(1,num_steps_pred):
        prev_new = np.delete(prev_new,0,1)
        prev_new = np.concatenate((prev_new,np.reshape(prediction[j-1],(1,1,7))),1)
        prediction[j] = model.predict(prev_new)
    prediction = np.transpose(prediction)
    return prediction


#Main function
def lstm():
    df = pd.read_csv(r'COVID-19_Combined_Mobility_And_Infection_Data_Moving_Avg_updated.csv')
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
    print(np.shape(data))


    #Create training and labels for lstm by sampling from longer sequence of 266 Days
    #takes 1000 samples of 20 day segments and the 21st day is the label
    TT_SPLIT = 240
    train_x, test_x = np.split(data, [TT_SPLIT], 1)

    #Trying splitting and training from 45 states, while testing with 5 states
    #train_x, test_x = np.split(data, [45], 0)

    #Get training data from sampling from larger time series
    train_sample_x, train_sample_y = sample(train_x, 3000, 21)
    #sample longer sequences to compare against predicted
    test_sample_x, test_sample_y = sample(test_x, 300, 36)


    #combine X and y to normalize and stationarize training
    combined = np.append(train_sample_x, train_sample_y, 1)

    #obtain min and max values for normalizing
    min_vals = np.ndarray.min(data,(0,1))
    max_vals = np.ndarray.max(data,(0,1))

    #stationarize data
    stat_data, start_point = stationary(combined)
    print(stat_data[0])
    print(np.shape(stat_data))

    #normalize data
    normalized_data, min, max = normalize(stat_data, min_vals, max_vals)
    print(normalized_data[0])
    print(np.shape(normalized_data))

    #obtain processed data by splitting again
    proc_train_x, proc_train_y = np.split(normalized_data, [20], 1)
    print(np.shape(proc_train_x))
    print(np.shape(proc_train_y))

    #Create model
    #Haven't tuned model at all.  Need to try different structures/parameters
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(20, input_shape=(len(proc_train_x[0]), 7), return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(20, return_sequences=False))
    model.add(tf.keras.layers.Dense(7))

    #compile model
    model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.001), metrics = ['accuracy'])

    print(model.summary())

    #if LOAD is true then load in previous model, else fit a new one
    if LOAD == True:
        model = tf.keras.models.load_model('lstm_recursive_model')
    else:
        model.fit(proc_train_x, proc_train_y, epochs=50, batch_size = 1, validation_split = 0.1)
        model.save('lstm_recursive_model')

    #combine X and y to normalize and stationarize testing
    combined_test = np.append(test_sample_x[0:100,:20], test_sample_y[:100,:20], 1)

    #stationarize here?
    stat_data_test, start_point_test = stationary(combined_test)

    #normalize testing data using min max found before
    normalized_data_test, min, max = normalize(stat_data_test, min_vals, max_vals)

    #obtain processed data by splitting again
    proc_test_x, proc_test_y = np.split(normalized_data_test, [20], 1)

    #num_graphs is how many graphs to print of predictions
    num_graphs = 20
    for i in range(0,num_graphs):
        #Create prediction from previous data
        prediction = predict(proc_test_x[i], 15, model)
        prediction = np.transpose(prediction)
        prediction = np.reshape(prediction, (1,15,7))


        #combine prediction with previous data to undo normalization/stationarizing
        combined_pred = np.append(proc_test_x[i], prediction)
        combined_pred = np.reshape(combined_pred, (1,35,7))

        #undo normalization
        undo_norm = undo_normalize(combined_pred, min, max)

        #undo stationarize here
        undo_stat = undo_stationary(undo_norm, start_point_test)
        diff = np.subtract(undo_stat[0][0][6],test_sample_x[i][0][6])
        for j in range(0,len(undo_stat[0])):
            undo_stat[0][j][6] = np.subtract(undo_stat[0][j][6],diff)
        diff = np.subtract(undo_stat[0][20][6],test_sample_x[i][20][6])
        for j in range(20,len(undo_stat[0])):
            undo_stat[0][j][6] = np.subtract(undo_stat[0][j][6],diff)
        #undo_stat[6] = np.add(undo_stat[6],test_sample_x[6][0])
        #undo_stat = np.transpose(undo_stat)

        #create plot of covid numbers prediction vs actual
        plot_final = np.transpose(undo_stat)
        plot_real = np.transpose(test_sample_x[i])
        plt.clf()
        plt.axvline(x=20,color='green', linestyle='dashed')
        plt.plot(plot_real[6])
        plt.plot(plot_final[6], '--')
        plt.savefig('prediction{0}.png'.format(i))

    #get mean value of # of cases off of actual value
    diff_sum = 0.0
    diff_base = 0.0
    for i in range(0,len(proc_test_x)):
        print(np.shape(proc_test_x[i]))
        prediction = predict(proc_test_x[i], 15, model)
        prediction = np.transpose(prediction)
        prediction = np.reshape(prediction, (1,15,7))
        undo_norm = undo_normalize(combined_pred, min, max)

        undo_stat = undo_stationary(undo_norm, start_point_test)
        #undo_stat = np.transpose(undo_stat)
        diff = np.subtract(undo_stat[0][0][6],test_sample_x[i][0][6])
        for j in range(0,len(undo_stat[0])):
            undo_stat[0][j][6] = np.subtract(undo_stat[0][j][6],diff)
        diff = np.subtract(undo_stat[0][20][6],test_sample_x[i][20][6])
        for j in range(20,len(undo_stat[0])):
            undo_stat[0][j][6] = np.subtract(undo_stat[0][j][6],diff)

        difference = abs(undo_stat[0,-1,6] - test_sample_x[i,-1,6])
        diff_sum += difference

        difference_base = abs(test_sample_x[i,19,6] - test_sample_x[i,-1,6])
        diff_base += difference_base
    print("average off: ")
    print(diff_sum / len(proc_test_x))
    print("base off: ")
    print(diff_base / len(proc_test_x))

    # #delete
    #Print predictions for full state test
    # for state_ind in range(0,51):
    #     stat_data_test, start_point_test = stationary(test_x)
    #     normalized_state, min, max = normalize(stat_data_test, min_vals, max_vals)
    #     predictions = np.zeros((len(test_x[state_ind])-35,36))
    #     plt.clf()
    #     for i in range(0,len(test_x[state_ind])-35):
    #         #proc_test = np.reshape(normalized_state[state_ind][i:i+20], (1,20,7))
    #         #prediction = model.predict(proc_test)
    #         #print(prediction)
    #         #predictions[i][0] = i+38
    #         #predictions[i][1] = prediction[0][6]
    #
    #         proc_test = np.reshape(normalized_state[state_ind][i:i+20], (1,20,7))
    #         print(np.shape(proc_test))
    #         proc_test = np.reshape(proc_test,(20,7))
    #         prediction = predict(proc_test, 15, model)
    #         prediction = np.transpose(prediction)
    #         prediction = np.reshape(prediction, (1,15,7))
    #
    #         #combine prediction with previous data to undo normalization/stationarizing
    #         combined_pred = np.append(proc_test_x[i], prediction)
    #         combined_pred = np.reshape(combined_pred, (1,35,7))
    #
    #         #undo normalization
    #         undo_norm = undo_normalize(combined_pred, min, max)
    #
    #         #undo stationarize here
    #         undo_stat = undo_stationary(undo_norm, start_point_test)
    #         diff = np.subtract(undo_stat[0][0][6],test_sample_x[i][0][6])
    #         for j in range(0,len(undo_stat[0])):
    #             undo_stat[0][j][6] = np.subtract(undo_stat[0][j][6],diff)
    #         diff = np.subtract(undo_stat[0][20][6],test_sample_x[i][20][6])
    #         for j in range(20,len(undo_stat[0])):
    #             undo_stat[0][j][6] = np.subtract(undo_stat[0][j][6],diff)
    #         for j in range(0,36):
    #             predictions[i][j] = undo_stat[0][j][6]
    #         print(predictions[i])
    #         x_val = np.arange(i,i+36)
    #         plt.plot(x_val, predictions[i])
    #     plt.savefig('testing.png')
    #
    #     #np.split(plot_real, [20], )
    #     #print(np.transpose(normalized_state[state_ind])[6])
    #     plt.plot(np.transpose(normalized_state[state_ind])[6])
    #     #plt.plot(plot_final[6][:20], '--')
    #     #plt.plot(plot_final[6][-1])
    #     predictions = np.transpose(predictions)
    #     plt.plot(predictions[0], predictions[1])
    #     plt.axvline(x=20,color='green', linestyle='dashed')
    #     plt.axvline(x=38,color='green', linestyle='dashed')
    #     #plt.plot(plot_prev[0],plot_prev[1])
    #     #plt.savefig('prediction.png')
    #     plt.savefig('state{0}.png'.format(states[state_ind]))


if __name__ == '__main__':
    lstm()
