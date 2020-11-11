import pandas as pd
import numpy as np
import copy


def create_dataset():
	#Load the csv data from our two different sources
	mobility_dataframe = pd.read_csv('2020_US_Region_Mobility_Report.csv', infer_datetime_format=True, parse_dates=True)
	cdc_dataframe = pd.read_csv('United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv', infer_datetime_format=True, parse_dates=True)

	#clean mobility data
	#get rid of the country_region_code, country_region, metro_area, iso_3166_2_code,census_fips_code
	mobility_dataframe=mobility_dataframe.drop(columns=['country_region_code', 'country_region', 'metro_area',  'iso_3166_2_code', 'census_fips_code'])
	#get rid of first few hundred rows, which are average across the entire US, where sub_region_1='nan'
	mobility_dataframe=mobility_dataframe.dropna(subset=['sub_region_1'])
	#because we want the average by state we need to get rid of rows which are split out by sub_region_2
	mobility_dataframe=mobility_dataframe[mobility_dataframe['sub_region_2'].isnull()]
	#now sub_region_2 is of no use, so we can remove it
	mobility_dataframe=mobility_dataframe.drop(columns=['sub_region_2'])

	#clean cdc data for sanity purposes - we are going to be interested in
	#total cases
	#new_cases
	cdc_dataframe=cdc_dataframe.drop(columns=['created_at', 'consent_cases', 'consent_deaths', 'prob_cases', 'prob_death', 'tot_death', 
		'conf_death','new_death','pnew_death', 'conf_cases'])
	#remove dates before the start of our mobility data
	cdc_dataframe=cdc_dataframe[cdc_dataframe['submission_date']>='02/15/2020']
	#remove dates after the end of our mobility data
	cdc_dataframe=cdc_dataframe[cdc_dataframe['submission_date']<='11/06/2020']
	#print(cdc_dataframe.max())
	#print(mobility_dataframe.max())
	#print(cdc_dataframe.shape)
	#print(mobility_dataframe.shape)

	#In mobility data states are listed by full list name, but in CDC data states are listed by abbreviation
	#Use the below dictionary to match
	state_dict = {
    	'Alabama': 'AL',
    	'Alaska': 'AK',
    	'American Samoa': 'AS',
    	'Arizona': 'AZ',
    	'Arkansas': 'AR',
    	'California': 'CA',
    	'Colorado': 'CO',
    	'Connecticut': 'CT',
    	'Delaware': 'DE',
    	'District of Columbia': 'DC',
    	'Florida': 'FL',
    	'Georgia': 'GA',
    	'Guam': 'GU',
    	'Hawaii': 'HI',
    	'Idaho': 'ID',
    	'Illinois': 'IL',
    	'Indiana': 'IN',
    	'Iowa': 'IA',
    	'Kansas': 'KS',
    	'Kentucky': 'KY',
    	'Louisiana': 'LA',
    	'Maine': 'ME',
    	'Maryland': 'MD',
    	'Massachusetts': 'MA',
    	'Michigan': 'MI',
    	'Minnesota': 'MN',
    	'Mississippi': 'MS',
    	'Missouri': 'MO',
    	'Montana': 'MT',
    	'Nebraska': 'NE',
    	'Nevada': 'NV',
    	'New Hampshire': 'NH',
    	'New Jersey': 'NJ',
    	'New Mexico': 'NM',
    	'New York': 'NY',
    	'North Carolina': 'NC',
    	'North Dakota': 'ND',
    	'Northern Mariana Islands':'MP',
    	'Ohio': 'OH',
    	'Oklahoma': 'OK',
    	'Oregon': 'OR',
    	'Pennsylvania': 'PA',
    	'Puerto Rico': 'PR',
    	'Rhode Island': 'RI',
    	'South Carolina': 'SC',
    	'South Dakota': 'SD',
    	'Tennessee': 'TN',
    	'Texas': 'TX',
    	'Utah': 'UT',
    	'Vermont': 'VT',
    	'Virgin Islands': 'VI',
    	'Virginia': 'VA',
    	'Washington': 'WA',
    	'West Virginia': 'WV',
    	'Wisconsin': 'WI',
    	'Wyoming': 'WY'
	}

if __name__ == '__main__':
	#Create dataset
	create_dataset()