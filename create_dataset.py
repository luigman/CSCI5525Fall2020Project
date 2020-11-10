import pandas as pd
import numpy as np
import copy


def create_dataset():
	#Load the csv data from our two different sources
	mobility_dataframe = pd.read_csv('2020_US_Region_Mobility_Report.csv')
	cdc_dataframe = pd.read_csv('United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv')

	#clean mobility data
	#get rid of the country_region_code, country_region, iso_3166_2_code,census_fips_code




if __name__ == '__main__':
	#Create dataset
	create_dataset()