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
LOAD = False

#make data stationary just the covid cases number or all the features?
#subtracts t-1 from t to create stationary data
def stationary(data):
    stationary_data = np.zeros(np.shape(data))
    stationary_data = np.delete(stationary_data, len(stationary_data[0])-1, 1)
    start_point = np.zeros((len(data), len(data[0][0])))
    for i in range(0,len(data)):
        start_point[i] = data[i][0]
        for j in range(1,len(data[0])):
            stationary_data[i][j-1] = np.subtract(data[i][j], data[i][j-1])
    return stationary_data, start_point

#normalize all of the features between -1 and 1
def normalize(data, min_vals, max_vals):
    #normalize between -1 and 1
    normalized_data = np.zeros(np.shape(data))
    for i in range(0,len(data)):
        for j in range(0,len(data[0])):
            #normalized_data[i][j] = (((data[i][j] - min_vals) / (max_vals - min_vals)) * 2) - 1
            normalized_data[i][j] = ((data[i][j] - min_vals) / (max_vals - min_vals))
    print(min_vals)
    print(max_vals)
    return normalized_data, min_vals, max_vals

#record initial starting point, add change to starting point for each timestep to rebuild data
def undo_stationary(data, start_point):
    print(np.shape(data))
    stationary_data = np.zeros((np.shape(data)[0], np.shape(data)[1] + 1, np.shape(data)[2]))
    print(np.shape(stationary_data))
    for i in range(0,len(data)):
        stationary_data[i][0] = start_point[i]
        for j in range(1,len(stationary_data[0])):
            stationary_data[i][j] = np.add(stationary_data[i][j-1], data[i][j-1])
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
    days_ahead = 14
    samples_x = np.zeros((num_samples,num_timesteps,7))
    samples_y = np.zeros((num_samples,1,7))
    for i in range(0,num_samples):
        state = random.randint(0,len(data)-1)
        day = random.randint(0,len(data[0])-(num_timesteps+1  +  days_ahead))
        samples_x[i] = data[state][day:day+num_timesteps]
        samples_y[i] = data[state][day+num_timesteps +  days_ahead]
    return samples_x, samples_y


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
    print(data[0])
    print(np.shape(data))



    #Create training and labels for lstm by sampling from longer sequence of 266 Days
    #takes 1000 samples of 20 day segments and the 21st day is the label
    #TT_SPLIT = 200
    #train_x, test_x = np.split(data, [TT_SPLIT], 1)
    train_x, test_x = np.split(data, [45], 0)

    train_sample_x, train_sample_y = sample(train_x, 3000, 20)
    #sample longer sequences to compare against predicted
    test_sample_x, test_sample_y = sample(test_x, 300, 35) #21
    print(np.shape(train_sample_x))
    print(np.shape(train_sample_y))

    #combine X and y to normalize and stationarize training
    combined = np.append(train_sample_x, train_sample_y, 1)
    print(np.shape(combined))

    min_vals = np.ndarray.min(combined,(0,1))
    max_vals = np.ndarray.max(combined,(0,1))

    normalized_data, min, max = normalize(combined, min_vals, max_vals)

    proc_train_x, proc_train_y = np.split(normalized_data, [20], 1)
    print("shapes")
    print(np.shape(proc_train_x))
    print(np.shape(proc_train_y))
    proc_train_y = proc_train_y[:len(proc_train_y), :len(proc_train_y[0]), 6]
    print(np.shape(proc_train_y))
    print(proc_train_y[0])

    #Create model
    #Haven't tuned model at all.  Need to try different structures/parameters
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(20, input_shape=(len(proc_train_x[0]), 7), return_sequences=False))
    #model.add(tf.keras.layers.Dropout(0.2))
    #model.add(tf.keras.layers.LSTM(20, return_sequences=False))
    model.add(tf.keras.layers.Dense(1)) #one output


    model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.001), metrics = ['accuracy'])

    print(model.summary())

    if LOAD == True:
        model = tf.keras.models.load_model('lstm_direct_model')
    else:
        model.fit(proc_train_x, proc_train_y, epochs=50, batch_size = 1, validation_split = 0.1)
        model.save('lstm_direct_model')

    #combine X and y to normalize and stationarize testing
    print("TESTING")
    combined_test = np.append(test_sample_x[0:100,:20], test_sample_y[:100,:20], 1)

    #stationary_data_test, start_point_test = stationary(combined_test)  #####

    normalized_data_test, min, max = normalize(combined_test, min_vals, max_vals)

    proc_test_x, proc_test_y = np.split(normalized_data_test, [20], 1)
    print(np.shape(proc_test_x))
    print(np.shape(proc_test_y))

    num_graphs = 10
    for i in range(0,num_graphs):
        prediction = predict(proc_test_x[i], 14, model)
        print(np.shape(prediction))
        prediction = np.reshape(prediction, (1,14,7))
        combined = np.append(proc_test_x[i], prediction)
        combined = np.reshape(combined, (1,34,7))
        print(np.shape(combined))
        undo_norm = undo_normalize(combined, min, max)
        #undo_stat = undo_stationary(undo_norm, start_point_test[i])
        print(undo_norm)


        plot_final = np.transpose(undo_norm)
        plot_real = np.transpose(test_sample_x[i])
        #plot_prev = np.transpose(prev_8)
        plt.clf()
        plt.plot(plot_real[4])
        plt.plot(plot_final[4], '--')
        #plt.plot(plot_prev[0],plot_prev[1])
        #plt.savefig('prediction.png')
        plt.savefig('prediction{0}.png'.format(i))



if __name__ == '__main__':
    lstm()
