import pandas as pd
import numpy as np
import datetime

def update_dataset():
	merge=pd.DataFrame()
	df_old = pd.read_csv('COVID-19_Combined_Mobility_And_Infection_Data_Moving_Avg.csv')
	df_new = pd.read_csv('COVID-19_Combined_Mobility_And_Infection_Data_Moving_Avg_updated.csv')

	for index, row in df_new.iterrows():
		match=df_old[(df_old['sub_region_1']==row['sub_region_1'])&(df_old['date']==row['date'])]
		if len(match)>0:
			merge=merge.append(match)
		else:
			merge=merge.append(row)

	df1 = merge[merge.isna().any(axis=1)]	
	merge=merge.interpolate(method='linear')

	merge.to_csv('COVID-19_Combined_Mobility_And_Infection_Data_Moving_Avg_updated_lin_int.csv', index=False)


if __name__ == '__main__':
	update_dataset()