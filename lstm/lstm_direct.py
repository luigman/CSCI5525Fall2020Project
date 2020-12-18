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
    start_point = np.zeros((len(data), len(data[0][0])))
    for i in range(0,len(data)):
        start_point[i] = data[i][0]
        for j in range(1,len(data[0])):
            stationary_data[i][j-1] = np.subtract(data[i][j], data[i][j-1])
    return stationary_data, start_point

#normalize all of the features between -1 and 1
def normalize(data, min_vals, max_vals):
    normalized_data = np.zeros(np.shape(data))
    for i in range(0,len(data)):
        for j in range(0,len(data[0])):
            normalized_data[i][j] = ((data[i][j] - min_vals) / (max_vals - min_vals))
    return normalized_data, min_vals, max_vals

#record initial starting point, add change to starting point for each timestep to rebuild data
def undo_stationary(data, start_point):
    stationary_data = np.zeros((np.shape(data)[0], np.shape(data)[1] + 1, np.shape(data)[2]))
    for i in range(0,len(data)):
        stationary_data[i][0] = start_point[i]
        for j in range(1,len(stationary_data[0])):
            stationary_data[i][j] = np.add(stationary_data[i][j-1], data[i][j-1])
    return stationary_data

def undo_normalize(norm_data, min_vals, max_vals):
    undo_normalized_data = np.zeros(np.shape(norm_data))
    for i in range(0,len(norm_data)):
        for j in range(0,len(norm_data[0])):
            undo_normalized_data[i][j] = ((norm_data[i][j]) * (max_vals - min_vals)) + min_vals
    return undo_normalized_data

#sample smaller sequence lengths from the longer ones to create more learnable data
def sample(data, num_samples, num_timesteps):
    days_ahead = 18
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
    df = pd.read_csv(r'COVID-19_Combined_Mobility_And_Infection_Data_Moving_Avg_updated_lin_int.csv')
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


    #Create training and labels for lstm by sampling from longer sequence of 266 Days
    #takes 1000 samples of 20 day segments and the 21st day is the label
    TT_SPLIT = 240
    train_x, test_x = np.split(data, [TT_SPLIT], 1)

    #Get training data from sampling from larger time series
    train_sample_x, train_sample_y = sample(train_x, 4000, 20)
    #sample longer sequences to compare against predicted
    test_sample_x, test_sample_y = sample(test_x, 300, 39)


    #combine X and y to normalize and stationarize training
    combined = np.append(train_sample_x, train_sample_y, 1)

    min_vals = np.ndarray.min(data,(0,1))
    max_vals = np.ndarray.max(data,(0,1))

    #normalize data
    normalized_data, min, max = normalize(combined, min_vals, max_vals)

    #obtain processed data by splitting again
    proc_train_x, proc_train_y = np.split(normalized_data, [20], 1)

    #because its training directly for final step, set all mobility values to 0 for
    #label so network only trains on case number
    for i in range(0,len(proc_train_y)):
        for j in range(0,len(proc_train_y[0])):
            proc_train_y[i][j] = [0,0,0,0,0,0,proc_train_y[i][j][6]]

    #Create model
    #Haven't tuned model at all.  Need to try different structures/parameters
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(20, input_shape=(len(proc_train_x[0]), 7), return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(10, return_sequences=False))
    model.add(tf.keras.layers.Dense(7))

    #compile model
    model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.001), metrics = ['accuracy'])

    print(model.summary())

    #if LOAD is true then load in previous model, else fit a new one
    LOAD = True
    if LOAD == True:
        model = tf.keras.models.load_model('lstm_direct_model')
    else:
        model.fit(proc_train_x, proc_train_y, epochs=100, batch_size = 1, validation_split = 0.1)
        model.save('lstm_direct_model')

    #combine X and y to normalize and stationarize testing
    combined_test = np.append(test_sample_x[0:100,:20], test_sample_y[:100,:20], 1)

    #normalize testing data using min max found before
    normalized_data_test, min, max = normalize(combined_test, min_vals, max_vals)

    #obtain processed data by splitting again
    proc_test_x, proc_test_y = np.split(normalized_data_test, [20], 1)


    #Generate plots for sample predictions
    print("GENERATING SAMPLE PREDICTION PLOTS")
    num_graphs = 10
    for i in range(0,num_graphs):
        proc_test = np.reshape(proc_test_x[i], (1,20,7))
        prediction = model.predict(proc_test)
        prediction = np.reshape(prediction,(1,1,7))
        undo_norm = undo_normalize(prediction, min, max)

        plot_real = np.transpose(test_sample_x[i])
        plt.clf()
        plt.title('LSTM Projected COVID case count')
        plt.xlabel('Time(days)')
        plt.ylabel('New cases per 100,000 people')

        plt.plot(plot_real[6])
        plt.axvline(x=20,color='green', linestyle='dashed')
        plt.plot([38], [undo_norm[0][0][6]], marker='o', markersize=3, color="red")
        plt.savefig('direct_prediction_sample_{0}.png'.format(i))


    #get mean value of # of cases off of actual value
    diff_sum = 0.0
    diff_base = 0.0
    for i in range(0,len(proc_test_x)):
        proc_test = np.reshape(proc_test_x[i], (1,20,7))
        prediction = model.predict(proc_test)
        prediction = np.reshape(prediction,(1,1,7))
        undo_norm = undo_normalize(prediction, min_vals, max_vals)


        difference = abs(undo_norm[0,0,6] - test_sample_x[i,-1,6])
        diff_sum += difference

        difference_base = abs(test_sample_x[i,19,6] - test_sample_x[i,-1,6])
        diff_base += difference_base
    print("PERCENT DEVIATION FROM TRUE VALUE: ")
    print(diff_sum / len(proc_test_x))



    # #Print predictions for full state test
    # normalized_state, min, max = normalize(test_x, min_vals, max_vals)
    # for state_ind in range(0,51):
    #     predictions = np.zeros((len(test_x[state_ind])-38,2))
    #     for i in range(0,len(test_x[state_ind])-38):
    #         proc_test = np.reshape(normalized_state[state_ind][i:i+20], (1,20,7))
    #         prediction = model.predict(proc_test)
    #         prediction = np.reshape(prediction,(1,1,7))
    #         prediction = undo_normalize(prediction, min_vals, max_vals)
    #         predictions[i][0] = i+38
    #         predictions[i][1] = prediction[0][0][6]
    #
    #     plt.clf()
    #     state = np.asarray(normalized_state)
    #     state = undo_normalize(state, min_vals, max_vals)
    #     plt.plot(np.transpose(state[state_ind])[6], label='Real Data')
    #     predictions = np.transpose(predictions)
    #     plt.plot(predictions[0], predictions[1], label='Predicted Future', marker='.', linestyle="None")
    #     plt.title('Direct LSTM Projected COVID case count')
    #     plt.xlabel('Time(days)')
    #     plt.ylabel('New Cases per 100,000 people')
    #     plt.axvline(x=20,color='green', linestyle='dashed')
    #     plt.axvline(x=38,color='green', linestyle='dashed')
    #     plt.legend(loc="upper left")
    #     plt.savefig('state{0}.png'.format(states[state_ind]))


if __name__ == '__main__':
    lstm()
