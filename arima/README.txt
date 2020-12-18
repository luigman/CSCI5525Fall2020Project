Instructions for running the ARIMA files
Requirement tensorflow

arima.py
Description: Runs the ARIMA+MLP models on a subset of the total data. From our testing, this subset is representative of the full dataset and results in identical plots as arima_multistate.
How to run:
python3 arima.py

inputs: None

outputs:
prints out average error for each time iteration
shows plot of ARIMA accuracy
optionally (if showplot > 0) shows the MLP prediction on top of the real and baseline data
optionally (if showplot > 1) shows the ARIMA projection of each mobility feature


arima_multistate.py
Description: Runs the ARIMA+MLP models on the full dataset. From our testing, this program produced identical plots to arima.py but takes much longer to run.
How to run:
python3 arima_multistate.py

inputs: None

outputs:
prints out average error for each time iteration
shows plot of ARIMA accuracy
optionally (if showplot > 0) shows the MLP prediction on top of the real and baseline data
optionally (if showplot > 1) shows the ARIMA projection of each mobility feature

We ran into an issue with some initializations of the MLP resulting in zero predicted cases at all times, but retraining the network seemed to fix these issues.
If you recieve the "Prediction is zero. Retraining..." error this is what is occurring.
In very rare cases where retraining the network does not fix the issue, the program will output "Could not train model on this data" and that month will be skipped.
