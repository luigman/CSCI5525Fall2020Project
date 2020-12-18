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

#normalize
def normalize(data, min_vals, max_vals):
    normalized_data = np.zeros(np.shape(data))
    for i in range(0,len(data)):
        for j in range(0,len(data[0])):
            #normalized_data[i][j] = (((data[i][j] - min_vals) / (max_vals - min_vals)) * 2) - 1
            normalized_data[i][j] = ((data[i][j] - min_vals) / (max_vals - min_vals))
    return normalized_data, min_vals, max_vals

#record initial starting point, add change to starting point for each timestep to rebuild data
def undo_stationary(data, start_point):
    stationary_data = np.zeros((np.shape(data)[0], np.shape(data)[1] + 1, np.shape(data)[2]))
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
    TT_SPLIT = 229
    train_x, test_x = np.split(data, [TT_SPLIT], 1)

    #Get training data from sampling from larger time series
    train_sample_x, train_sample_y = sample(train_x, 3000, 21)
    #sample longer sequences to compare against predicted
    test_sample_x, test_sample_y = sample(test_x, 100, 71)


    #combine X and y to normalize and stationarize training
    combined = np.append(train_sample_x, train_sample_y, 1)

    #obtain min and max values for normalizing
    min_vals = np.ndarray.min(data,(0,1))
    max_vals = np.ndarray.max(data,(0,1))

    #stationarize data
    stat_data, start_point = stationary(combined)

    #normalize data
    normalized_data, min, max = normalize(stat_data, min_vals, max_vals)

    #obtain processed data by splitting again
    proc_train_x, proc_train_y = np.split(normalized_data, [20], 1)

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
    LOAD = True
    if LOAD == True:
        model = tf.keras.models.load_model('lstm_recursive_model')
    else:
        model.fit(proc_train_x, proc_train_y, epochs=500, batch_size = 1, validation_split = 0.1)
        model.save('lstm_recursive_model')

    #combine X and y to normalize and stationarize testing
    combined_test = np.append(test_sample_x[0:300,:20], test_sample_y[:300,:20], 1)

    #stationarize here
    stat_data_test, start_point_test = stationary(combined_test)

    #normalize testing data using min max found before
    normalized_data_test, min, max = normalize(stat_data_test, min_vals, max_vals)

    #obtain processed data by splitting again
    proc_test_x, proc_test_y = np.split(normalized_data_test, [20], 1)

    #num_graphs is how many graphs to print of predictions
    print("GENERATING SAMPLE PREDICTION PLOTS")
    num_graphs = 10
    for i in range(0,num_graphs):
        #Create prediction from previous data
        prediction = predict(proc_test_x[i], 30, model)
        prediction = np.transpose(prediction)
        prediction = np.reshape(prediction, (1,30,7))


        #combine prediction with previous data to undo normalization/stationarizing
        combined_pred = np.append(proc_test_x[i], prediction)
        combined_pred = np.reshape(combined_pred, (1,50,7))

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

        #create plot of covid numbers prediction vs actual
        plot_final = np.transpose(undo_stat)
        plot_real = np.transpose(test_sample_x[i][:50])
        plt.clf()
        plt.title('LSTM Projected COVID case count')
        plt.xlabel('Time(days)')
        plt.ylabel('New cases per 100,000 people')

        plt.axvline(x=20,color='green', linestyle='dashed')
        plt.plot(plot_real[6], label = 'Real Data')
        plt.plot(np.arange(20,50),plot_final[6][20:-1], '--', label = 'Predicted Future')
        plt.legend(loc="upper left")
        plt.savefig('recursive_prediction_sample_{0}.png'.format(i))

    #Get accuracy values for predicting 1-50 out and plot as rec_lstm_accuracy
    print("GENERATING PERCENT DEVIATION FROM TRUE VALUES FOR PREDICTING 1-50 STEPS")
    diff_sum = 0.0
    diff_base = 0.0
    total_perc = 0.0
    divide_zero = 0
    percent_pred = np.zeros(50)
    divide_zero_count = np.zeros(50)
    for i in range(0,len(proc_test_x)):
        prediction = predict(proc_test_x[i], 50, model)
        prediction = np.transpose(prediction)
        prediction = np.reshape(prediction, (1,50,7))
        undo_norm = undo_normalize(combined_pred, min, max)
        undo_stat = undo_stationary(undo_norm, start_point_test)
        diff = np.subtract(undo_stat[0][0][6],test_sample_x[i][0][6])
        for j in range(0,len(undo_stat[0])):
            undo_stat[0][j][6] = np.subtract(undo_stat[0][j][6],diff)
        diff = np.subtract(undo_stat[0][20][6],test_sample_x[i][20][6])
        for j in range(20,len(undo_stat[0])):
            undo_stat[0][j][6] = np.subtract(undo_stat[0][j][6],diff)

        for j in range(0,50):
            if test_sample_x[i,-(50-j),6] == 0:
                divide_zero_count[j] += 1
            else:
                difference = abs(undo_stat[0,-(50-j),6] - test_sample_x[i,-(50-j),6])
                percent = difference/abs(test_sample_x[i,-1,6])
                percent_pred[j] += percent
    den = np.full(50,len(proc_test_x))
    den = den - divide_zero_count
    results = percent_pred/den
    print("Percent deviation from true values: 1-50 steps ahead")
    print(results)

    plt.clf()
    plt.title('Recursive LSTM Projection Accuracy')
    plt.xlabel('Days in advanced to predict')
    plt.ylabel('Percent deviation from true value')

    plt.plot(results)
    plt.savefig('rec_lstm_accuracy.png')


if __name__ == '__main__':
    lstm()
