import seaborn as sns
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def histograms_numeric_columns(df, numerical_columns):
    """
    Takes df, numerical columns as list
    Returns a group of histagrams
    """
    f = pd.melt(df, value_vars=numerical_columns)
    g = sns.FacetGrid(f, col="variable", col_wrap=4, sharex=False, sharey=False)
    g = g.map(sns.distplot, "value")
    return g


# Plot multiple columns seaborn
def plot_multiple_columns(data):
    df = data[columns]
    n = len(df.columns)
    fig, ax = plt.subplots(1, n, figsize=(12, n * 2), sharex=True)
    for i in range(n):
        plt.sca(ax[i])
        col = df.columns[i]
        sns.countplot(x=None, y=df[col].values, data=df)
        plt.title(f"Title based on {col}")

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    # jitter = [[False, 1], [0.5, 0.2]]

    for j in range(len(ax)):
        for i in range(len(ax[j])):
            ax[j][i].tick_params(labelsize=15)
            ax[j][i].set_xlabel("label", fontsize=17, position=(0.5, 20))
            ax[j][i].set_ylabel("label", fontsize=17)
            # x as add obstacle distance
            ax[j][i] = sns.stripplot(
                x="Sex", y="SidestepDist", jitter=jitter[j][i], data=daten_csv, ax=ax[j][i]
            )
    fig.suptitle("Categorical Features Overview", position=(0.5, 1.1), fontsize=20)
    fig.tight_layout()
    fig.show()

def visualize_null_values():
    sns.heatmap(df.isnull(),yticklabels=False)
    plt.show()

# Plots for country comparison per product
def plot_time_series_data_in_cluster(ts_data,country_cluster,image_dump,dataset_name,experiment="experiment"):
  for c in country_cluster.cluster.unique():
    cluster = country_cluster[country_cluster.cluster==c]
    country_list = cluster['names'].unique()
    product = pd.pivot_table(ts_data,index='year',columns='country_product',
                            values='export_val',aggfunc=np.mean)
    df = product[country_list]
    df.fillna(0,inplace=True)
    plt.figure(figsize=(10,5.5))
    df.plot(subplots=False,figsize=(10,5.5),legend=False,
                              title=(f"Exports for countries in cluster {c} from {dataset_name} for {experiment}"))
    plt.xticks(rotation=70)
    # plt.legend()
    plt.ylabel("Export value")
    plt.savefig(f"{image_dump}/{experiment}_{dataset_name}_cluster_{c}")
    plt.show()