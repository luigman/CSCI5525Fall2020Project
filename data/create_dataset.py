import pandas as pd
import numpy as np
import datetime


def create_dataset(avg):
    #Load the csv data from our two different sources
    mobility_df = pd.read_csv('2020_US_Region_Mobility_Report_dec.csv')
    cdc_df = pd.read_csv('United_States_COVID-19_Cases_and_Deaths_by_State_over_Time_dec.csv')

    #clean mobility data
    #get rid of the country_region_code, country_region, metro_area, iso_3166_2_code,census_fips_code
    mobility_df=mobility_df.drop(columns=['country_region_code', 'country_region', 'metro_area',  'iso_3166_2_code', 'census_fips_code'])
    #get rid of first few hundred rows, which are average across the entire US, where sub_region_1='nan'
    mobility_df=mobility_df.dropna(subset=['sub_region_1'])
    #because we want the average by state we need to get rid of rows which are split out by sub_region_2
    mobility_df=mobility_df[mobility_df['sub_region_2'].isnull()]
    #now sub_region_2 is of no use, so we can remove it
    mobility_df=mobility_df.drop(columns=['sub_region_2'])
    #let's change our dates to a datetime for future comparisons
    mobility_df['date']=pd.to_datetime(mobility_df['date'], format='%Y-%m-%d')

    #let's also change our dates in cdc data to a datetime for future comparisons
    cdc_df['submission_date']=pd.to_datetime(cdc_df['submission_date'], format='%m/%d/%Y')

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

    #We have to normalize the data by state population
    #We're using the US Census Bureau data 2019 estimates
    state_pop_df=pd.read_csv('SCPRC-EST2019-18+POP-RES.csv')

    #Build y_value column
    new_cases_per_state_per_day=[]
    for index, row in mobility_df.iterrows():
        date=row['date']
        state=state_dict[row['sub_region_1']]

        #get state population, then divide by 100,000, since we'll regulate by per 100,000 people in the state
        state_pop_filtered=state_pop_df[state_pop_df['NAME']==row['sub_region_1']]
        if(len(state_pop_filtered)!=1):
            print("HELP! I'm having issues filtering for the state population")
        state_pop=float(state_pop_filtered['POPESTIMATE2019'].fillna(0))/100000

        if(avg):
            num_cases=0
            match = cdc_df[(cdc_df['submission_date']>(date-datetime.timedelta(days=7)))&(cdc_df['submission_date']<=date)&(cdc_df['state']==state)]
            match=match.fillna(0)
            for i, r in match.iterrows():
                num_cases+=float(r['new_case'])+float(r['pnew_case'])
            new_cases_per_state_per_day.append(num_cases/(7*state_pop))
        else:
            match=cdc_df[(cdc_df['submission_date']==date)&(cdc_df['state']==state)]
            if(len(match)==1):
                new_cases_per_state_per_day.append((float(match['new_case'].fillna(0))+float(match['pnew_case'].fillna(0)))/state_pop)
            else:
                new_cases_per_state_per_day.append(np.nan)

    mobility_df['num_cases']=new_cases_per_state_per_day

    if(avg):
        mobility_df.to_csv('COVID-19_Combined_Mobility_And_Infection_Data_Moving_Avg_updated_test.csv', index=False)
    else:
        mobility_df.to_csv('COVID-19_Combined_Mobility_And_Infection_Data_updated.csv', index=False)


if __name__ == '__main__':
    #Create dataset
    create_dataset(True)
    create_dataset(False)