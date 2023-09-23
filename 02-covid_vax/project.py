# project.py


import numpy as np
import pandas as pd
import os

# If this import errors, run `pip install plotly` in your Terminal with your conda environment activated.
import plotly.express as px


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def count_monotonic(arr):
    return np.sum(arr[1:] < arr[:-1])

def monotonic_violations_by_country(vacs): 
    mono_group = vacs.groupby('Country_Region').agg({
        'Doses_admin': lambda x: count_monotonic(np.array(x)),
        'People_at_least_one_dose': lambda x: count_monotonic(np.array(x))
    })
    mono_group.columns = ['Doses_admin_monotonic', 'People_at_least_one_dose_monotonic']
    return mono_group 
    


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def robust_totals(vacs):
    grouped = vacs.groupby('Country_Region')
    
    doses_admin = grouped['Doses_admin'].quantile(0.97)
    people_vaccinated = grouped['People_at_least_one_dose'].quantile(0.97)

    result = pd.DataFrame({
            'Doses_admin': doses_admin,
            'People_at_least_one_dose': people_vaccinated
    })
    return result


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def fix_dtypes(pops_raw):
    pops_df = pops_raw.copy()
    pops_df['Population in 2023'] = (pops_df['Population in 2023']*1000).astype(int)
    pops_df['World Percentage'] = pops_df['World Percentage'].apply(lambda x: float(x.replace('%',''))/100)
    pops_df['Area (Km²)'] = pops_df['Area (Km²)'].replace(regex=[r'\D+'], value="")#map(lambda i: ''.join([x for x in i[:-1] if x.isdigit()]))##.map(lambda x: int(''.join([i for i in x if i.isnumeric()][:-1])))
    pops_df['Density (P/Km²)'] = pops_df['Density (P/Km²)'].apply(lambda x: float(x[:-4]))
    return pops_df


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def missing_in_pops(tots, pops):
    return set(tots.index) - set(pops['Country (or dependency)'])

    
def fix_names(pops):
    pops_fixed = pops.copy()
    pops_fixed.loc[pops_fixed['Country (or dependency)'] == 'Myanmar', 'Country (or dependency)'] = 'Burma'
    pops_fixed.loc[pops_fixed['Country (or dependency)'] == 'Cape Verde', 'Country (or dependency)'] = 'Cabo Verde'
    pops_fixed.loc[pops_fixed['Country (or dependency)'] == 'Republic of the Congo', 'Country (or dependency)'] = 'Congo (Brazzaville)'
    pops_fixed.loc[pops_fixed['Country (or dependency)'] == 'DR Congo', 'Country (or dependency)'] = 'Congo (Kinshasa)'
    pops_fixed.loc[pops_fixed['Country (or dependency)'] == 'Ivory Coast', 'Country (or dependency)'] = "Cote d\'Ivoire"
    pops_fixed.loc[pops_fixed['Country (or dependency)'] == 'Czech Republic', 'Country (or dependency)'] = 'Czechia'
    pops_fixed.loc[pops_fixed['Country (or dependency)'] == 'South Korea', 'Country (or dependency)'] = 'Korea, South'
    pops_fixed.loc[pops_fixed['Country (or dependency)'] == 'United States', 'Country (or dependency)'] = 'US'
    pops_fixed.loc[pops_fixed['Country (or dependency)'] == 'Palestine', 'Country (or dependency)'] = 'West Bank and Gaza'
    return pops_fixed


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def draw_choropleth(tots, pops_fixed):
    merged_df = tots.merge(pops_fixed, left_index=True, right_on='Country (or dependency)')
    doses_per_person = merged_df['Doses_admin'] / merged_df['Population in 2023']


    data = pd.DataFrame({'Country': merged_df['Country (or dependency)'],
                            'Doses Per Person': doses_per_person, 'ISO': merged_df['ISO']})
    fig = px.choropleth(data_frame=data,
                            locations=merged_df['ISO'],#merged_df['Country (or dependency)'],
                            #locationmode='country names',
                            color='Doses Per Person',
                            range_color=(0, 4),
                            color_continuous_scale='YlGnBu',
                            hover_name='Country',
                            #hover_data={'Doses Per Person': ':.3f'},
                            labels={'Doses Per Person': 'Doses Per Person', 'ISO': 'ISO'},
                            title='COVID Vaccine Doses Per Person')
    fig.update_layout(title_font_family='Arial', title_font_size=24)
    return fig


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_israel_data(df):
    israel_clean = df.copy()
    
    israel_clean['Vaccinated'] = israel_clean['Vaccinated'].astype(bool)

    israel_clean['Severe Sickness'] = israel_clean['Severe Sickness'].astype(bool)

    israel_clean['Age'] = pd.to_numeric(israel_clean['Age'], errors='coerce')

    return israel_clean


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def mcar_permutation_tests(df, n_permutations):
    
    def calc_test_stat(column):
        missing = column[df['Age'].isna()]
        not_missing = column[~df['Age'].isna()]
        
        return abs(np.mean(missing) - np.mean(not_missing))
    
    vax_missing = df['Vaccinated'][df['Age'].isna()]
    vax_not_missing = df['Vaccinated'][~df['Age'].isna()]
    obs_stat = calc_test_stat(df['Vaccinated'])
    perm_stats = []
    ss_missing = df['Severe Sickness'][df['Age'].isna()]
    ss_not_missing = df['Severe Sickness'][~df['Age'].isna()]
    obs_stat2 = calc_test_stat(df['Severe Sickness'])
    perm_stats2 = []
    
    for i in range(n_permutations):
        perm_vax = np.random.permutation(np.concatenate([vax_missing, vax_not_missing]))
        perm_stat = calc_test_stat(perm_vax)
        perm_stats.append(perm_stat)
        
        perm_ss = np.random.permutation(np.concatenate([ss_missing, ss_not_missing]))
        perm_stat2 = calc_test_stat(perm_ss)
        perm_stats2.append(perm_stat2)
    
    return np.array(perm_stats), np.array(perm_stats2)
    
    
def missingness_type():
    return 1


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def effectiveness(df):
    pv = df.loc[df['Vaccinated'], 'Severe Sickness'].mean()
    
    pu = df.loc[~df['Vaccinated'], 'Severe Sickness'].mean()
    
    eff = 1 - pv / pu
    
    return eff


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


AGE_GROUPS = [
    '12-15',
    '16-19',
    '20-29',
    '30-39',
    '40-49',
    '50-59',
    '60-69',
    '70-79',
    '80-89',
    '90-'
]

def stratified_effectiveness(df):

    age12_15 = df[(df['Age'] >= 12) & (df['Age'] <= 15)]
    age16_19 = df[(df['Age'] >= 16) & (df['Age'] <= 19)]
    age20_29 = df[(df['Age'] >= 20) & (df['Age'] <= 29)]
    age30_39 = df[(df['Age'] >= 30) & (df['Age'] <= 39)]
    age40_49 = df[(df['Age'] >= 40) & (df['Age'] <= 49)]
    age50_59 = df[(df['Age'] >= 50) & (df['Age'] <= 59)]
    age60_69 = df[(df['Age'] >= 60) & (df['Age'] <= 69)]
    age70_79 = df[(df['Age'] >= 70) & (df['Age'] <= 79)]
    age80_89 = df[(df['Age'] >= 80) & (df['Age'] <= 89)]
    age90_ = df[(df['Age'] >= 90)]

    age12_15_pv = age12_15[(age12_15['Vaccinated'] == True) & (age12_15['Severe Sickness'] == True)].shape[0] / age12_15[(age12_15['Vaccinated'] == True)].shape[0]
    age12_15_pu = age12_15[(age12_15['Vaccinated'] == False) & (age12_15['Severe Sickness'] == True)].shape[0] / age12_15[(age12_15['Vaccinated'] == False)].shape[0]
    age12_15_eff = 1-(age12_15_pv/age12_15_pu)

    age16_19_pv = age16_19[(age16_19['Vaccinated'] == True) & (age16_19['Severe Sickness'] == True)].shape[0] / age16_19[(age16_19['Vaccinated'] == True)].shape[0]
    age16_19_pu = age16_19[(age16_19['Vaccinated'] == False) & (age16_19['Severe Sickness'] == True)].shape[0] / age16_19[(age16_19['Vaccinated'] == False)].shape[0]
    age16_19_eff = 1-(age16_19_pv/age16_19_pu)

    age20_29_pv = age20_29[(age20_29['Vaccinated'] == True) & (age20_29['Severe Sickness'] == True)].shape[0] / age20_29[(age20_29['Vaccinated'] == True)].shape[0]
    age20_29_pu = age20_29[(age20_29['Vaccinated'] == False) & (age20_29['Severe Sickness'] == True)].shape[0] / age20_29[(age20_29['Vaccinated'] == False)].shape[0]
    age20_29_eff = 1-(age20_29_pv/age20_29_pu)

    age30_39_pv = age30_39[(age30_39['Vaccinated'] == True) & (age30_39['Severe Sickness'] == True)].shape[0] / age30_39[(age30_39['Vaccinated'] == True)].shape[0]
    age30_39_pu = age30_39[(age30_39['Vaccinated'] == False) & (age30_39['Severe Sickness'] == True)].shape[0] / age30_39[(age30_39['Vaccinated'] == False)].shape[0]
    age30_39_eff = 1-(age30_39_pv/age30_39_pu)

    age40_49_pv = age40_49[(age40_49['Vaccinated'] == True) & (age40_49['Severe Sickness'] == True)].shape[0] / age40_49[(age40_49['Vaccinated'] == True)].shape[0]
    age40_49_pu = age40_49[(age40_49['Vaccinated'] == False) & (age40_49['Severe Sickness'] == True)].shape[0] / age40_49[(age40_49['Vaccinated'] == False)].shape[0]
    age40_49_eff = 1-(age40_49_pv/age40_49_pu)

    age50_59_pv = age50_59[(age50_59['Vaccinated'] == True) & (age50_59['Severe Sickness'] == True)].shape[0] / age50_59[(age50_59['Vaccinated'] == True)].shape[0]
    age50_59_pu = age50_59[(age50_59['Vaccinated'] == False) & (age50_59['Severe Sickness'] == True)].shape[0] / age50_59[(age50_59['Vaccinated'] == False)].shape[0]
    age50_59_eff = 1-(age50_59_pv/age50_59_pu)

    age60_69_pv = age60_69[(age60_69['Vaccinated'] == True) & (age60_69['Severe Sickness'] == True)].shape[0] / age60_69[(age60_69['Vaccinated'] == True)].shape[0]
    age60_69_pu = age60_69[(age60_69['Vaccinated'] == False) & (age60_69['Severe Sickness'] == True)].shape[0] / age60_69[(age60_69['Vaccinated'] == False)].shape[0]
    age60_69_eff = 1-(age60_69_pv/age60_69_pu)

    age70_79_pv = age70_79[(age70_79['Vaccinated'] == True) & (age70_79['Severe Sickness'] == True)].shape[0] / age70_79[(age70_79['Vaccinated'] == True)].shape[0]
    age70_79_pu = age70_79[(age70_79['Vaccinated'] == False) & (age70_79['Severe Sickness'] == True)].shape[0] / age70_79[(age70_79['Vaccinated'] == False)].shape[0]
    age70_79_eff = 1-(age70_79_pv/age70_79_pu)

    age80_89_pv = age80_89[(age80_89['Vaccinated'] == True) & (age80_89['Severe Sickness'] == True)].shape[0] / age80_89[(age80_89['Vaccinated'] == True)].shape[0]
    age80_89_pu = age80_89[(age80_89['Vaccinated'] == False) & (age80_89['Severe Sickness'] == True)].shape[0] / age80_89[(age80_89['Vaccinated'] == False)].shape[0]
    age80_89_eff = 1-(age80_89_pv/age80_89_pu)

    age90_pv = age90_[(age90_['Vaccinated'] == True) & (age90_['Severe Sickness'] == True)].shape[0] / age90_[(age90_['Vaccinated'] == True)].shape[0]
    age90_pu = age90_[(age90_['Vaccinated'] == False) & (age90_['Severe Sickness'] == True)].shape[0] / age90_[(age90_['Vaccinated'] == False)].shape[0]
    age90_eff = 1-(age90_pv/age90_pu)

    age90_eff

    strat_eff = pd.Series(
        data=[age12_15_eff, age16_19_eff, age20_29_eff, age30_39_eff, age40_49_eff, age50_59_eff, age60_69_eff,
            age70_79_eff, age80_89_eff, age90_eff] 
        , index=AGE_GROUPS
        )
    
    return strat_eff


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def effectiveness_calculator(
    *,
    young_vaccinated_prop,
    old_vaccinated_prop,
    young_risk_vaccinated,
    young_risk_unvaccinated,
    old_risk_vaccinated,
    old_risk_unvaccinated
):
     
    overall_risk_vac = (young_risk_vaccinated * young_vaccinated_prop) + (old_risk_vaccinated * old_vaccinated_prop)
    overall_risk_unvac = ((1 - young_vaccinated_prop) * young_risk_unvaccinated) + ((1 - old_vaccinated_prop) * old_risk_unvaccinated)
    
    overall_eff = 1 - (overall_risk_vac / overall_risk_unvac)
    
    young_eff = 1 - (young_risk_vaccinated / young_risk_unvaccinated)
    
    old_eff = 1 - (old_risk_vaccinated / old_risk_unvaccinated)
    
    return {'Overall': overall_eff, 
            'Young': young_eff, 
            'Old': old_eff}


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def extreme_example():
    return {
        'young_vaccinated_prop': 0.56,
        'old_vaccinated_prop': 0.99,
        'young_risk_vaccinated': 0.01,
        'young_risk_unvaccinated': 0.20,
        'old_risk_vaccinated': 0.10,
        'old_risk_unvaccinated': 0.50
    }
