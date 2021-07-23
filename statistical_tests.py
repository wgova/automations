from scipy.stats import shapiro, norm, kstest
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller

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

def adf_test_multiple_columns(df,variable_name):
  r=df.apply(lambda x: adfuller(x,autolag='AIC')).T
  r.columns = [f'{variable_name}_adf_test_statistic',
               f'{variable_name}_adf_p-value',
               f'{variable_name}_adf_lags_used',
               f'{variable_name}_adf_obs',
               f'crit_vals','x']
  crits = r['crit_vals'].apply(pd.Series)
  crits.columns = [f'{variable_name}_adf_1%_crit',
                   f'{variable_name}_5%_adf_crit',
                   f'{variable_name}_10%_adf_crit']
  d=pd.concat([r.drop(['crit_vals','x'], axis=1),crits],axis=1)
  print(f'Completed tests for {variable_name}')
  return d

def kpss_test_multiple_columns(df,variable_name):
    r=df.apply(lambda x: kpss(x,lags="auto",regression='ct')).T
    r.columns = [f'{variable_name}_kpss_test_statistic',
                 f'{variable_name}_kpss_p-value',
                 f'{variable_name}_kpss_lags_used',
                 f'crit_vals']
    crits = r['crit_vals'].apply(pd.Series)
    crits.columns = [f'{variable_name}_adf_10%_crit',
                   f'{variable_name}_5%_adf_crit',
                   f'{variable_name}_2.5%_adf_crit',
                   f'{variable_name}_1%_adf_crit']
    d=pd.concat([r.drop(['crit_vals'], axis=1),crits],axis=1)
    print(f'Completed tests for {variable_name}')
    return d

def apply_stationarity_test_to_df_dict(dict_name,test):
  list_df =[]
  for key,values in dict_name.items():
    if test=='adf':
      df=adf_test_multiple_columns(dict_name[key],variable_name=key)
    else:
      df=kpss_test_multiple_columns(dict_name[key],variable_name=key)
    list_df.append(df)
  return list_df
# https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
# https://towardsdatascience.com/6-ways-to-test-for-a-normal-distribution-which-one-to-use-9dcf47d8fa93
# https://medium.com/@rrfd/testing-for-normality-applications-with-python-6bf06ed646a9
