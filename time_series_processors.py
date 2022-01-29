import time
import numpy as np
import pandas as pd
df['filled_col'] = df.groupby(['colA','colB'])['targetcol'].transform(
    lambda grp: grp.fillna(np.mean(grp))
)


Group by names_category and reindex to the whole date range
# Define helper function
def add_missing_years(grp):
    _ = grp.set_index('Year')
    _ = _.reindex(list(range(2005,2019)))
    del _['names_category']
    return _
# Group by country name and extend
df = df.groupby('names_category').apply(add_missing_years)
df = df.reset_index()

Interpolate for years between and extrapolate for years outside the range for 
which we have observations on a per-country basis
# Define helper function
def fill_missing(grp):
    res = grp.set_index('Year')\
    .interpolate(method='linear',limit=5)\
    .fillna(method='ffill')\
    .fillna(method='bfill')
    del res['names_category']
    return res
# Group by names_category and fill missing
df = df.groupby(['names_category']).apply(
    lambda grp: fill_missing(grp)
)

def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek + 1
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4  
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)  
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)   
    return df