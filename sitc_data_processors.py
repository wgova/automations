import numpy as np
import pandas as pd
from functools import reduce
import logging
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import re
import pathlib

import statsmodels.api as sm
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller

import tsfresh
from tsfresh import extract_features, select_features
from tsfresh import defaults
from tsfresh.feature_extraction import settings,feature_calculators
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from tsfresh.utilities import dataframe_functions, profiling
from tsfresh.utilities.distribution import MapDistributor, MultiprocessingDistributor,DistributorBaseClass
from tsfresh.utilities.string_manipulation import convert_to_output_format
from tsfresh.feature_extraction.settings import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import roll_time_series
from kats.tsfeatures.tsfeatures import TsFeatures,TimeSeriesData
from kats.detectors.outlier import MultivariateAnomalyDetector, MultivariateAnomalyDetectorType
from kats.models.var import VARParams
from kats.detectors.cusum_detection import CUSUMDetector
from kats.detectors.trend_mk import MKDetector
from kats.consts import TimeSeriesData
from kats.utils.simulator import Simulator
from kats.tsfeatures.tsfeatures import TsFeatures
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
import nolds #Hurst hypothesis

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


def prefix_origin_to_sitc(df,country_code_column,sitc_column):
    df['exporter'] = df[[country_code_column,sitc_column]].apply(
        func=(lambda row: '_'.join(row.values.astype(str))), axis=1)
    return df

def extract_ts_features(df,time_col, name_col,value_col,feature_calculator):
    features = extract_features(df[[time_col, name_col,value_col]],
                     column_id=name_col, column_sort=time_col,
                     column_kind=None, column_value=None,
                     default_fc_parameters=feature_calculator)
    return features

def extract_tsfresh_kats_features(countries:list,country_dict:dict,min_feats,path_to_data):
    df_list = []
    size = range(len(countries))
    for c,country in zip(size,countries):#countries:
        target = f'{path_to_data}/features/{country}_features.csv'
        file = pathlib.Path(target)
        if file.exists():
            print (f"Features dataset for {country} already exist - feature extraction skipped")
        else:
            df = country_dict[country]
            pivot = pd.pivot(df,index='year',columns='exporter',values='export_value')\
            .reset_index()\
            .fillna(0)
            # Transform data to time series object
            pivot.rename(columns={'year':'time','export_value':'value'},inplace=True)
            pivot['time'] = pd.to_datetime(pivot.time,format='%Y')
            ts_features = extract_ts_features(df,'year','exporter','export_value',min_feats)
            cols = [col for col in pivot if col not in ['time']]#.iloc[:,1:].columns
            list_ts = []
            for i in tqdm(range(len(cols)),"Products completed:"):
                data =  pivot[['time',cols[i]]]
                ts = TimeSeriesData(data,date_format='%Y')
                del data
                # Initialise TSFeatures model object
                model = TsFeatures(selected_features=['cusum_detector','hw_params','nowcasting',
                                                    'holt_params','seasonalities','trend_detector',
                                                    'statistics','trend_strength','stl_features'])
                # Generate kats features
                feat = model.transform(ts)
                feat['exporter'] = cols[i]
                list_ts.append(feat)
                del ts
            try:
                features = pd.DataFrame(list_ts)
                kat_features = features.set_index('exporter')
                del features
                all_feat = pd.concat(
                    [ts_features,kat_features],axis=0)
                all_feat.to_csv(target)
                print (f"Feature extraction for {country} completed")
                display(f"{c} out of {len(countries)}: {100*(c/len(countries)):.0f}/%\ completed")
            except:
                print('something wrong')
            # df_list.append(all_feat)
  