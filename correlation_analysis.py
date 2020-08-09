import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

# ----------------------- Correlation analyses-------------------------------------------
def heatmap_numeric_w_dependent_variable(df, dependent_variable):
    """
    Takes df, a dependant variable as str
    Returns a heatmap of all independent variables' correlations with dependent variable 
    """
    plt.figure(figsize=(10, 5.5))
    figure = sns.heatmap(
        df.corr()[[dependent_variable]].sort_values(by=dependent_variable),
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    return figure


def plot_correlated_features(df, threshold: float):
    corr = df.corr()
    colour_limits = corr[corr >= threshold]
    plt.figure(figsize=(10, 5.5))
    sns.heatmap(colour_limits, cmap="Greens")
    plt.title(f"Features with correlation above {threshold*100}%")

# Also try ideas from https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf
def drop_correlated_pairs(df, threshold: float):
    corr_array  = df.corr().abs()\
        .unstack()
    sorted_corr_array = corr_array.\
        sort_values(kind="quicksort", ascending=False).drop_duplicates()
    sorted_corr = pd.DataFrame(sorted_corr_array).\
        reset_index().\
            rename(columns={"level_0": "feature_1", 
                "level_1": "feature_2", 
                0: "score"},
                inplace=True,)
    collinear_array = sorted_corr[sorted_corr["score"] >= threshold]
    exclude_collinear_feats = collinear_array["feature_2"].values
    # use correlations above 0.5
    # bug: excluding singular pairs with r=1
    return exclude_collinear_feats

# selected_columns = selected_columns[1:].values
def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return x, columns
# Example usage:
# SL = 0.05
# data_modeled, selected_columns = backwardElimination(data.iloc[:,1:].values, data.iloc[:,0].values, SL, selected_columns)

# TODO
# Augmented Dickey-Fuller
