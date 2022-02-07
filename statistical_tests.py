from scipy.stats import shapiro, norm, kstest
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from .data_processors import merge_df_list

# normality test
def shapiro_wilk_test(data, alpha=0.05):
    """
    H0: sample was drawn from a Gaussian distribution
    For a series x in data, calculates the Shapiro-Wilk statistic
    Return: statistic and p-value
    """
    statistic = []
    p_value = []
    sw_test = []
    decision = []
    for col in data.columns:
        stat, p = shapiro(data[col].values)
        # print("Statistics=%.2f, p=%.2f" % (stat, p))
        statistic.append(stat)
        p_value.append(p)
        # interpret
        alpha = 0.05
        if p > alpha:
            decision.append("Gaussian")
            # print("Sample likely Gaussian, fail to reject H0)")
        else:
            decision.append("Not Gaussian")
        # print("Sample likely not Gaussian (reject H0)")
    sw_test = pd.concat(
        [statistic, p_value, decision],
        axis=1,
        keys=["statistic", "p_value", "Decision"],
    )
    return sw_test

def adf_test_ts_columns(df,variable_name):
  r=df.apply(lambda x: adfuller(x.fillna(method='ffill'),autolag='AIC')).T
  r.columns = [f'{variable_name}_adf_test_statistic',
               f'{variable_name}_adf_p-value',
               f'{variable_name}_adf_lags_used',
               f'{variable_name}_adf_obs',
               'crit_vals','x']
  crits = r['crit_vals'].apply(pd.Series)
  crits.columns = [f'{variable_name}_adf_1%_crit',
                   f'{variable_name}_adf_5%_crit',
                   f'{variable_name}_adf_10%_crit']
  d=pd.concat([r.drop(['crit_vals','x'], axis=1),crits],axis=1)
  print(f'Completed tests for {variable_name}')
  return d

def kpss_test_ts_columns(df,variable_name):
    try:
      #Differenced time series have negative values therefore infinite intermediates that cannot be converted to int during 'auto' lag
        a=df.apply(lambda x: kpss(x.fillna(method='ffill'),nlags="legacy",regression='c')).T
    except ValueError:
            pass
    a.columns = [f'{variable_name}_kpss_test_statistic',
                 f'{variable_name}_kpss_p-value',
                 f'{variable_name}_kpss_lags_used',
                 'crit_vals']
    crits = a['crit_vals'].apply(pd.Series)
    crits.columns = [f'{variable_name}_kpss_10%_crit',
                     f'{variable_name}_kpss_5%_crit',
                     f'{variable_name}_kpss_2_5%_crit',
                     f'{variable_name}_kpss_1%_crit']
    d=pd.concat([a.drop(['crit_vals'], axis=1),crits],axis=1)
    print(f'Completed tests for {variable_name}')
    return d

def test_stationarity_for_dict_of_dfs(dict_name,test,category_name):
  list_valid_tests = ['adf','kpss']
  if test not in list_valid_tests:
        raise ValueError('Valid tests for this function are "adf" and "kpss"')
  list_df =[]
  validated = ['export_value','log_export_value','diff1_export_value','log_diff1_export_value','diff2_export_value']
  for k,v in dict_name.items():
    print(f'Running tests for: {k}')
    if k in validated:
      test_df = dict_name[k]
      if test=='adf':
          df=adf_test_ts_columns(test_df,variable_name=k)
      else:
          df=kpss_test_ts_columns(test_df,variable_name=k)
    else:
      print(f'{k} not specified for {test}')
    list_df.append(df)
  print(f'Output: len{list_df[0]}')
  return merge_df_list(list_df,category_name)
# https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
# https://towardsdatascience.com/6-ways-to-test-for-a-normal-distribution-which-one-to-use-9dcf47d8fa93
# https://medium.com/@rrfd/testing-for-normality-applications-with-python-6bf06ed646a9
