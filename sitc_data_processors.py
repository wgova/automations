import numpy as np
import pandas as pd

def sitc_codes_to_sectors(df,sitc_code_column:'int64'):
    '''
    Allocate sectors to SITC codes
    sitc_code_column: integer codes
    '''
    conditions=[df[sitc_code_column]<990,
            df[sitc_code_column]<1223,
            df[sitc_code_column]<2930,
            df[sitc_code_column]<3511,
            df[sitc_code_column]<4315,
            df[sitc_code_column]<6000,
            df[sitc_code_column]<7000,
            df[sitc_code_column]<7940,
            df[sitc_code_column]<9000,
            df[sitc_code_column]<9710]
    choices = ['food_animals','beverages_tobacco','crude_materials_inedibles','mineral_fuels_lubricants',
    'animals_vegetable_oils','chemicals_products','manufactured_goods','machinery_and_transport',
    'miscellaneous_manufactured_articles','commodities_and_other_transactions']
    df['sectors'] = np.select(conditions,choices,default='miscellaneous_unspecified')
    return df

def country_name_changes(df,sitc_code_column):
    '''
    Country name changes sometime not updated across all databases. This list is inexhaustive. 
    '''
    dict_names = {'ddr':'deu','fdr':'deu','sun':'rus','csk':'cze','ymd':'yem','yar':'yem'}
    if len(df[df.origin.isin(dict_names.keys())]) != 0:
        df[sitc_code_column]= df[sitc_code_column].map(dict_names)\
            .fillna(df[sitc_code_column])
        print("Multiple country codes fixed")
    else:
        print("No country codes fixed")