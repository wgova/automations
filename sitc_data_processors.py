import numpy as np
import pandas as pd
from functools import reduce

def sitc_codes_to_sectors(df,sitc_code_column):
    '''
    Allocate sectors to SITC codes
    sitc_code_column: integer codes
    sector_0: food_animals - 09
    sector_1: beverages_tobacco - 12
    sector_2:'crude_materials_inedibles',29
    sector_3':'mineral_fuels_lubricants',35
    sector_4:'animals_vegetable_oils',43
    sector_5:'chemicals_products',59
    sector_6:'manufactured_goods',69
    sector_7:'machinery_and_transport',79
    sector_8:'miscellaneous_manufactured_articles',89
    sector_9:'commodities_and_other_transactions',97
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
            df[sitc_code_column]<9710,
            df[sitc_code_column] == 'travel',
            df[sitc_code_column] == 'transport', 
            df[sitc_code_column] == 'ict', 
            df[sitc_code_column] == 'financial']
    choices = ['food_animals','beverages_tobacco','crude_materials_inedibles','mineral_fuels_lubricants',
    'animals_vegetable_oils','chemicals_products','manufactured_goods','machinery_and_transport',
    'miscellaneous_manufactured_articles','commodities_and_other_transactions','travel',
    'transport', 'ict', 'financial']
    df['sectors'] = np.select(conditions,choices,default='unspecified')
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

def transform_data(dframe):
    '''
    input: pivot export data, indexed by year
    output: log transformed and differenced data sets
    '''
    x_log = dframe.fillna(1)
    x_diff = dframe.fillna(0)
    log_data = np.log(x_log)\
        .reset_index()\
        .melt(id_vars='year',var_name = 'exporter',value_name='log_export_value')
    diff1_data = x_diff.diff(periods=1)\
        .reset_index()\
        .melt(id_vars='year',var_name = 'exporter',value_name='diff1_export_value')
    diff2_data = x_diff.diff(periods=2)\
        .reset_index()\
        .melt(id_vars='year',var_name = 'exporter',value_name='diff2_export_value')
    log_diff1_data = np.log(x_log).diff(periods=1)\
        .reset_index()\
        .melt(id_vars='year',var_name = 'exporter',value_name='log_diff1_export_value')
    exports_data = dframe\
        .reset_index()\
        .melt(id_vars='year',var_name = 'exporter',value_name='export_value')
    frames = [exports_data,log_data,diff1_data,diff2_data,log_diff1_data]
    transforms_merged = reduce(lambda  left,right: pd.merge(
        left,right,on=['year','exporter'],how='outer'), frames)
    return transforms_merged

def combine_country_code(df,country_code_column,sitc_column):
    df['exporter'] = df[country_code_column,sitc_column].apply(
        func=(lambda row: '_'.join(row.values.astype(str))), axis=1)
    return df