import scipy.stats as scipy
from scipy import shapiro

# normality test
def shapiro_wilk_test(data, variable_to_test=None, alpha=0.5):
    if variable_to_test == None:
        stat, p = shapiro(data)
        print("Statistics=%.3f, p=%.3f" % (stat, p))
    else:
        stat, p = shapiro(data.variable_to_test)
    # interpret
    alpha = 0.05
    if p > alpha:
        print("Sample looks Gaussian (fail to reject H0)")
    else:
        print("Sample does not look Gaussian (reject H0)")
