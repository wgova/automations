import seaborn as sns

# ----------------------- Correlation analyses-------------------------------------------
def heatmap_numeric_w_dependent_variable(df, dependent_variable):
    """
    Takes df, a dependant variable as str
    Returns a heatmap of all independent variables' correlations with dependent variable 
    """
    plt.figure(figsize=(8, 10))
    figure = sns.heatmap(
        df.corr()[[dependent_variable]].sort_values(by=dependent_variable),
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    return figure


def plot_correlated_features(df, threshold=0.5):
    corr = df.corr()
    colour_limits = corr[corr >= threshold]
    fig.suptitle(f"Features with correlation above {threshold*100}%")
    sns.heatmap(colour_limits, cmap="Greens")


def drop_correlated_pairs(df, threshold=0.5):
    corr = df.corr().abs()
    corr_array = corr.unstack()
    sorted_corr_array = corr_array.sort_values(
        kind="quicksort", ascending=False
    ).drop_duplicates()
    sorted_corr = pd.DataFrame(sorted_corr_array).reset_index()
    sorted_corr.rename(
        columns={"level_0": "feature_1", "level_1": "feature_2", 0: "score"},
        inplace=True,
    )
    collinear_array = sorted_corr[sorted_corr["score"] >= threshold]
    exclude_collinear_feats = collinear_array["feature_2"].values
    return exclude_collinear_feats
