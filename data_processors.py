import glob
import re
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import Binarizer, LabelEncoder
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

# test dataset
# data = pd.read_csv("sample_ts_data.csv")

def clean_header(df):
    """
	Removes name spaces, bracketscharacters and spaces from column names 
    Makes all header names lower case
	"""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace('"', "")
        .str.replace("'", "")
    )
    return df


def change_column_prefix(df, old_prefix="export_val__", new_prefix=None):
    if new_prefix == None:
        df.columns = [re.sub(f"^{old_prefix}", "", x) for x in df.columns]
    else:
        df.columns = [re.sub(f"^{old_prefix}", new_prefix, x) for x in df.columns]
    return df


def simplify_column_name(df, old_name, new_name):
    df.columns = df.columns = [re.sub(f"^{old_name}", f"{new_name}", x) for x in df.columns]
    return df.columns


def get_date_int(df, date_column):
    year = df[date_column].dt.year
    month = df[date_column].dt.month
    week = df[date_column].dt.week
    return year, month, week


def days_diff(df):
    df["days_since"] = (df["date_col2"] - df["date_col1"]).dt.days


def calculate_time_difference(df, date_col2, date_col1):
    """
	date_col2: name of the column with the signup date
	date_col1: name of the column with the last login
	"""
    event_year, event_month, event_week = get_date_int(df, event_date)
    lastevent_year, lastevent_month, lastevent_week = get_date_int(
        df, lastevent_activity
    )
    years_diff = lastevent_year - sign_year
    months_diff = lastlogin_month - sign_month
    weeks_diff = lastlogin_week - sign_week
    df["first_group"] = years_diff * 52 + weeks_diff + 1
    df["second_group"] = years_diff * 12 + months_diff + 1


# def mass_edit(file_prefix, folder_path='',col_to_change=None,operation=None):
#     """
#     file_prefix: string that defines new file name
#     folder_path: string copied from file explorer to the folder where the files are
# 	"""
#     if folder_path == '':
#         folder_path = input('Please enter path to CSV files:\n')
#     folder_path = folder_path.replace("\\","/")
#     if folder_path[:-1] != "/":
#         folder_path = folder_path + "/"
#     file_list = glob.glob(folder_path + '*.csv')
# 	for file in file_list:
#         name_pos = file.rfind('\\')
# 		data = pd.read_csv(file)
# 		data.to_csv(os.path.join(folder_path,file[name_pos+1:],file_prefix), index=False) #saving the file again with same name
# 		print(f'{file} ready!!')
#         return None

# -------------------------Missing values----------------------------
def table_percent_nulls_(df):
    """
    Create table showing percent null values if columns of dataframe
    """
    if df.isnull().sum().sum() > 0:
        mask_total = df.isnull().sum().sort_values(ascending=False)
        total = mask_total[mask_total > 0]
        mask_percent = df.isnull().mean().sort_values(ascending=False)
        percent = 100 * mask_percent[mask_percent > 0]
        missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
        print(f"Total and Percentage of NaN:\n {missing_data}")
    else:
        print("No NaN found.")


def check_null_columns(df, missing_percent):
    """
    Checks which columns have over specified percentage of missing values
    Takes df, missing percentage
    Returns columns as a list
    """
    mask_percent = df.isnull().mean()
    series = mask_percent[mask_percent > missing_percent]
    columns = series.index.to_list()
    return columns


def drop_null_columns(df, missing_percent):
    """
    Takes df, missing percentage
    Drops the columns whose missing value is bigger than missing percentage
    Returns df
    """
    series = check_null_columns(df, missing_percent=missing_percent)
    list_of_cols = series.index.to_list()
    df.drop(columns=list_of_cols)
    return df


# Alternatively, set a threshold for nans
def remove_null_values(df, threshold: int = 0.8):
    pct_null = df.isnull().sum() / len(df)
    missing_features = pct_null[pct_null > threshold].index
    df.drop(missing_features, axis=1, inplace=True)
    df.fillna(0, inplace=True)
    return df


def impute_mean(df):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp_mean.fit(df)
    return df


###-------------------- Outliers-----------------------------------------------
def check_outliers_iqr(df):
    """
    Takes df with continous variables,
    Checks nulls using Tukey's method 
    Reference: https://www.kdnuggets.com/2017/01/3-methods-deal-outliers.html
    """
    col = list(df)
    outliers = pd.DataFrame(columns=["columns", "Outliers"])
    for column in col:
        if column in df.select_dtypes(include=np.number).columns:
            # TODO: check if variables are continuous and normally distributed
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            below = q1 - (1.5 * q3 - q1)
            above = q3 + (1.5 * q3 - q1)
            outliers = outliers.append(
                {
                    "columns": column,
                    "outliers": df.loc[
                        (df[column] < below) | (df[column] > above)
                    ].shape[0],
                },
                ignore_index=True,
            )
    return outliers


# function for removing outliers
def remove_outliers_winsorize(dataframe):
    cols = dataframe.columns
    for col in cols:
        if col in dataframe.select_dtypes(include=np.number).columns:
            dataframe[col] = winsorize(
                dataframe[col], limits=[0.1, 0.1], inclusive=(True, True)
            )

    return dataframe

def remove_outliers_iqr(data):
  cols = data.columns
  for col in cols:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    IQR = q3 - q1
    threshold = 1.5 * IQR
    data = data[data[col].between((q1 - q3), (q3 + threshold))]
  return data

def outliers_zscore(data):
    cols = data.columns
    for col in cols:
        threshold = 3
        mean = data[col].mean()
        stdev = data[col].std()
        data[col_zscores] = (data[col] - mean) / stdev # compute zscore
        data = data[abs(data[col_zscores])<=threshold] # remove outliers
        data = data.drop(col_zscores, axis=1) # drop zscore columns
    return data 

# separate the num columns and cat data columns
# TODO
def get_categorical_columns(df):
    num_cols = df.select_dtypes(exclude=["object", "category"]).columns
    cat_cols = [i for i in df.columns if i not in df[num_cols].columns]
    for i in cat_cols:
        df[i] = df[i].astype("category")
    return cat_cols


def rename_column_variables(dataframe, col_name):
    dataframe[col_name].map(
        {"current_first": "new_first", "current_second": "new_second"}
    )
    return dataframe


# transform categorical features into numerical (ordinal) features:
# Step 1 - output a function, i.e. a transformer, that will transform
# each str in a list into a int, where the int is the index of that element
# in the list.
# Step 2 - ingest a dictionary and turn it into a a tranformer to map onto dataframe


def categorical_to_ordinal_transformer(categories):
    """
    Returns a function that will map categories to ordinal values based on the
    order of the list of `categories` given. Ex.
    If categories is ['A', 'B', 'C'] then the transformer will map 
    'A' -> 0, 'B' -> 1, 'C' -> 2.
    """
    return lambda categorical_value: categories.index(categorical_value)


def transform_categorical_to_numercial(df, categorical_numerical_mapping):
    """
    Transforms categorical columns to numerical columns
    Takes a df, a dictionary 
    Returns df
    """
    transformers = {
        k: categorical_to_ordinal_transformer(v)
        for k, v in categorical_numerical_mapping.items()
    }
    new_df = df.copy()
    for col, transformer in transformers.items():
        new_df[col] = new_df[col].map(transformer).astype("int64")
    return new_df


def scale_features(df):
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(
        scaler.fit_transform(df), columns=df.columns, index=df.index
    )
    return scaled_df


# Inspired by:
# https://towardsdatascience.com/creating-python-functions-for-exploratory-data-analysis-and-data-cleaning-2c462961bd71
# https://towardsdatascience.com/automate-boring-tasks-with-your-own-functions-a32785437179
# https://www.datacamp.com/community/tutorials/moving-averages-in-pandas
# https://stackabuse.com/scikit-learn-save-and-restore-models/
# https://stackabuse.com/tensorflow-neural-network-tutorial/\
# https://queirozf.com/entries/pandas-dataframe-groupby-examples#group-by-and-change-aggregation-column-name
