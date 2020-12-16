import pandas as pd
import numpy as np


def update_datset():
	df = pd.read_csv('COVID-19_Combined_Mobility_And_Infection_Data_Moving_Avg_updated_test.csv')

	states=df['sub_region_1'].unique()

	for s in states:
		df_at_state=df[df['sub_region_1']==s]
		for c in df_at_state:
			if(df_at_state[c].isnull().values.any()):
				df_at_state[c].fillna(df_at_state[c].mode()[0], inplace=False)

	for c in df:
		if(df[c].isnull().values.any()):
			df[c].fillna(df[c].mode()[0], inplace=True)


	df.to_csv('COVID-19_Combined_Mobility_And_Infection_Data_Moving_Avg_updated_mode.csv', index=False)


if __name__ == '__main__':
	update_datset()