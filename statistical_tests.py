from scipy.stats import shapiro, norm, kstest
import pandas as pd


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


# https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
# https://towardsdatascience.com/6-ways-to-test-for-a-normal-distribution-which-one-to-use-9dcf47d8fa93
# https://medium.com/@rrfd/testing-for-normality-applications-with-python-6bf06ed646a9
