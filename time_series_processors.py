import time
import numpy as np
import pandas as pd
from functools import reduce


#Group by names_category and reindex to the whole date range
# Define helper function
def add_missing_years(grp,datecolumn,min_date,max_date):
    _ = grp.set_index(datecolumn)
    _ = _.reindex(list(range(min_date,max_date)))
    del _['names_category']
    return _
# Group by country name and extend
# df = df.groupby('names_category').apply(add_missing_years)
# df = df.reset_index()

# Interpolate for years between and extrapolate for years outside the range for 
# which we have observations on a per-country basis
# Define helper function
def fill_missing(grp,datecolumn,fill_limit):
    res = grp.set_index(datecolumn)\
    .interpolate(method='linear',limit=fill_limit)\
    .fillna(method='ffill')\
    .fillna(method='bfill')
    del res['names_category']
    return res
# Group by names_category and fill missing
# df = df.groupby(['names_category']).apply(
#     lambda grp: fill_missing(grp)
# )

# df['filled_col'] = df.groupby(['colA','colB'])['targetcol'].transform(
#     lambda grp: grp.fillna(np.mean(grp))
# )

def create_date_features(df):
    df['month'] = df.date.dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek + 1
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4  
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)  
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    df['rolling_mean_7'] = df[target].shift(7).rolling(window=7).mean()
    df['lag_7'] = df[target].shift(7)
    df['lag_15']=df[target].shift(15)
    df['lag_last_year']=df[target].shift(52).rolling(window=15).mean()   
    return df

def interpretable_tsfresh_features():
    dict_features = {'abs_energy':None,
                 'first_location_of_maximum':None,
                 'first_location_of_minimum':None,
                 'last_location_of_maximum':None,
                 'last_location_of_minimum':None,
                 'absolute_sum_of_changes':None,
                 'percentage_of_reoccurring_datapoints_to_all_datapoints':None,
                 'percentage_of_reoccurring_values_to_all_values':None,
                 'ratio_value_number_to_time_series_length':None,
                 'sum_of_reoccurring_data_points': None,
                 'sum_of_reoccurring_values': None,
                 'time_reversal_asymmetry_statistic': [{'lag': 1}, {'lag': 2}, {'lag': 3}],
                 'median':None, 'mean':None,'standard_deviation':None,
                 'quantile':[{'q':0.1},{'q':0.2},{'q':0.8},{'q':0.9}],
                 'benford_correlation':None,
                 'autocorrelation':[{'lag': 0},{'lag': 1}, {'lag': 2}, {'lag': 3},{'lag': 4},{'lag': 5}],
                 'partial_autocorrelation':[{'lag': 1}, {'lag': 2}, {'lag': 3},{'lag': 4},{'lag': 5}],
                 'ar_coefficient': [{'coeff': 0, 'k': 10}, {'coeff': 1, 'k': 10},{'coeff': 2, 'k': 10}, {'coeff': 3, 'k': 10}],
                 'c3':[{'lag': 1}, {'lag': 2}, {'lag': 3}],
                 'cid_ce':[{'normalize': True}, {'normalize': False}],
                 'agg_autocorrelation': [{'f_agg': 'mean', 'maxlag': 40}, 
                                         {'f_agg': 'median', 'maxlag': 40}, {'f_agg': 'var', 'maxlag': 40}],
                 'binned_entropy':[{'max_bins': 10}],
                 'approximate_entropy':[{'m': 2, 'r': 0.1}, {'m': 2, 'r': 0.3}, 
                                        {'m': 2, 'r': 0.5}, {'m': 2, 'r': 0.7}, {'m': 2, 'r': 0.9}],
                 'linear_trend_timewise': [{'attr': 'pvalue'}, {'attr': 'rvalue'},{'attr': 'slope'}, {'attr': 'stderr'}],
                 'linear_trend':[{'attr': 'pvalue'}, {'attr': 'rvalue'}, {'attr': 'intercept'}, {'attr': 'slope'}, {'attr': 'stderr'}],
                 'lempel_ziv_complexity': [{'bins': 2}, {'bins': 3}, {'bins': 5}, {'bins': 10}],
                 'fourier_entropy': [{'bins': 2}, {'bins': 3}, {'bins': 5}, {'bins': 7}, {'bins': 10}],
                 'permutation_entropy':[{'tau': 1, 'dimension': 3}, {'tau': 1, 'dimension': 5}, 
                                        {'tau': 1, 'dimension': 7},{'tau': 1, 'dimension': 10}],
                 'root_mean_square': None,
                 'sample_entropy':None,
                 'kurtosis':None,
                 'skewness':None,
                 'median':None,
                 'standard_deviation':None,
                 'variance':None,
                 'fourier_entropy':[{'bins': 2}, {'bins': 3}, {'bins': 5}, {'bins': 10}],
                 'number_peaks': [{'n': 1}, {'n': 3}, {'n': 5}, {'n': 10}],
                 'number_cwt_peaks': [{'n': 1}, {'n': 5}],'number_peaks': [{'n': 1}, {'n': 3}, {'n': 5}, {'n': 10}], 
                 'cwt_coefficients':[{'coeff': 0, 'w': 2, 'widths': (2, 5, 10, 20)},
                                     {'coeff': 0, 'w': 5, 'widths': (2, 5, 10, 20)},
                                     {'coeff': 0, 'w': 10, 'widths': (2, 5, 10, 20)},
                                     {'coeff': 0, 'w': 20, 'widths': (2, 5, 10, 20)},
                                     {'coeff': 1, 'w': 2, 'widths': (2, 5, 10, 20)},
                                     {'coeff': 1, 'w': 5, 'widths': (2, 5, 10, 20)},
                                     {'coeff': 1, 'w': 10, 'widths': (2, 5, 10, 20)},
                                     {'coeff': 1, 'w': 20, 'widths': (2, 5, 10, 20)}],
                 'fft_aggregated':[{'aggtype': 'centroid'},
                                   {'aggtype': 'variance'},
                                   {'aggtype': 'skew'},
                                   {'aggtype': 'kurtosis'}],
                 'fft_coefficient':[{'attr': 'real', 'coeff': 0},{'attr': 'real', 'coeff': 1},{'attr': 'real', 'coeff': 2},
                                    {'attr': 'real', 'coeff': 3},{'attr': 'real', 'coeff': 4},{'attr': 'real', 'coeff': 5},
                                    {'attr': 'real', 'coeff': 6},{'attr': 'real', 'coeff': 7},{'attr': 'real', 'coeff': 8},
                                    {'attr': 'real', 'coeff': 9},{'attr': 'real', 'coeff': 10},{'attr': 'abs', 'coeff': 0},
                                    {'attr': 'imag', 'coeff': 0},{'attr': 'imag', 'coeff': 1},{'attr': 'imag', 'coeff': 2},
                                    {'attr': 'imag', 'coeff': 3},{'attr': 'imag', 'coeff': 4},{'attr': 'imag', 'coeff': 5},
                                    {'attr': 'imag', 'coeff': 6},{'attr': 'imag', 'coeff': 7},{'attr': 'imag', 'coeff': 8},
                                    {'attr': 'imag', 'coeff': 9},{'attr': 'imag', 'coeff': 10},{'attr': 'abs', 'coeff': 1},
                                    {'attr': 'abs', 'coeff': 2},{'attr': 'abs', 'coeff': 3},{'attr': 'abs', 'coeff': 4},
                                    {'attr': 'abs', 'coeff': 5},{'attr': 'abs', 'coeff': 6},{'attr': 'abs', 'coeff': 7},
                                    {'attr': 'abs', 'coeff': 8},{'attr': 'abs', 'coeff': 9},{'attr': 'abs', 'coeff': 10},
                                    {'attr': 'angle', 'coeff': 0},{'attr': 'angle', 'coeff': 1},{'attr': 'angle', 'coeff': 2},
                                    {'attr': 'angle', 'coeff': 3},{'attr': 'angle', 'coeff': 4},{'attr': 'angle', 'coeff': 5},
                                    {'attr': 'angle', 'coeff': 6},{'attr': 'angle', 'coeff': 7},{'attr': 'angle', 'coeff': 8},
                                    {'attr': 'angle', 'coeff': 9},{'attr': 'angle', 'coeff': 10}],
                 'spkt_welch_density': [{'coeff': 2}, {'coeff': 5}, {'coeff': 8}],
                 'friedrich_coefficients': [{'coeff': 0, 'm': 3, 'r': 30},{'coeff': 1, 'm': 3, 'r': 30},
                                            {'coeff': 2, 'm': 3, 'r': 30},{'coeff': 3, 'm': 3, 'r': 30}],
                 'max_langevin_fixed_point': [{'m': 3, 'r': 30}],
                 'change_quantiles':[{'qh':0.8,'isabs':False,'f_agg':'median','ql':0.0},{'qh':0.2,'ql':0.0,'isabs':False,'f_agg':'median'}],
                 'variation_coefficient': None
                 }
    return dict_features

def detrend_ts_data(dframe,datecolumn,category,values):
    '''
    #TODO : make more dynamic
    input: pivot export data, indexed by year
    output: log transformed and differenced data sets
    '''
    x_log = dframe.fillna(1)
    x_diff = dframe.fillna(0)
    log_data = np.log(x_log)\
        .reset_index()\
        .melt(id_vars=datecolumn,var_name = category,value_name=f'log_{values}')
    diff1_data = x_diff.diff(periods=1)\
        .reset_index()\
        .melt(id_vars=datecolumn,var_name = category,value_name=f'diff1_{values}')
    diff2_data = x_diff.diff(periods=2)\
        .reset_index()\
        .melt(id_vars=datecolumn,var_name = category,value_name=f'diff2_{values}')
    log_diff1_data = np.log(x_log).diff(periods=1)\
        .reset_index()\
        .melt(id_vars=datecolumn,var_name = category,value_name=f'log_diff1_{values}')
    exports_data = dframe\
        .reset_index()\
        .melt(id_vars=datecolumn,var_name = category,value_name=f'{values}')
    frames = [exports_data,diff1_data,diff2_data,log_data,log_diff1_data]
    transforms_merged = reduce(lambda  left,right: pd.merge(
        left,right,on=[datecolumn,category],how='outer'), frames)
    transforms_merged = transforms_merged[transforms_merged['{values}'].notna()]
    transforms_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    transforms_merged=transforms_merged.fillna(0)
    return transforms_merged