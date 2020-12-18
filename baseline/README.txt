Instructions for running the baseline files
Requirement tensorflow

baseline_full_data_shift_by_state.py
Description: Runs the baseline and POC models on the full dataset. Takes ~2 hours to run due to the quantity of models being trained.
How to run:
python3 baseline_full_data_shift_by_state.py

inputs: showPlot - toggles whether plots are shown

outputs:
shows optimal offset graphs for linear and logarithmic correlations
prints out optimal offset
shows accuracy graphs for baseline and POC models


baseline.py
Description: runs the baseline and POC models on only Minnesota data.
How to run:
python3 baseline.py

inputs: showPlot - toggles whether plots are shown

outputs:
shows optimal offset graphs for linear and logarithmic correlations
prints out optimal offset
shows accuracy graphs for baseline and POC models
