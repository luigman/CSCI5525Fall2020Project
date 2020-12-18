Instructions for running the LSTM files
Requirement tensorflow

lstm_recursive.py
How to run:
python3 lstm_recursive.py

inputs: None

outputs:
prints out lstm architecture
prints out percent deviation from true value for predictions 1-50 time-steps
saves plots for 10 example predictions
saves plot for percent deviation vs num steps predicted


lstm_direct.py
How to run:
python3 lstm_direct.py

inputs: None

outputs:
prints out lstm architecture
prints out percent deviation from true value for 18 days ahead
saves plots for 10 example predictions


Adding the following lines to the beginning of the file fixed "Fail to find the dnn implementation." error on one of our group member's computers.
If you do not recieve this error then they are not necessary.

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
