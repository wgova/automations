from scipy.stats import shapiro, norm,kstest


# normality test
def shapiro_wilk_test(data, variable_to_test=None, alpha=0.5):

    """
    For a series x in data, calculates the Shapiro-Wilk statistic
    Return: statistic and p-value
    """
    if variable_to_test == None:
        stat, p = shapiro(data)
        print("Statistics=%.2f, p=%.2f" % (stat, p))
    else:
        stat, p = shapiro(data[variable_to_test])
    # interpret
    alpha = 0.05
    if p > alpha:
        print("Sample likely Gaussian, fail to reject H0)")
    else:
        print("Sample does not look Gaussian (reject H0)")

# https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
# https://towardsdatascience.com/6-ways-to-test-for-a-normal-distribution-which-one-to-use-9dcf47d8fa93
# https://medium.com/@rrfd/testing-for-normality-applications-with-python-6bf06ed646a9
