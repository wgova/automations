import numpy as np
import pandas as pd
from functools import reduce
import logging
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import re
import pathlib
from .feature_selection import *

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
            df[sitc_code_column]<9710]
    choices = ['food_animals','beverages_tobacco','crude_materials_inedibles','mineral_fuels_lubricants',
    'animals_vegetable_oils','chemicals_products','manufactured_goods','machinery_and_transport',
    'miscellaneous_manufactured_articles','commodities_and_other_transactions']
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


def prefix_origin_to_sitc(df,country_code_column,sitc_column):
    df['exporter'] = df[[country_code_column,sitc_column]].apply(
        func=(lambda row: '_'.join(row.values.astype(str))), axis=1)
    return df

# def filter_features(features_df, features_to_find):
#     feature_list = []
#     for i in features_to_find:
#         feature_cols = features_df.filter(regex=i, axis=1).columns.to_list()
#         feature_list.extend(feature_cols)
#     return feature_list

def create_synth_ts():
    range = pd.date_range(start= 1960, periods=50,freq='A')
    series = pd.to_datetime(range, infer_datetime_format=True)
    data = pd.DataFrame(series, columns=['year'])
    data['exporter'] = 'IMG_110'
    data['export_value'] = np.random.randint(1500000, 3000000, size=(len(series)))
    # data = data.set_index('year')
    return data