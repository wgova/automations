﻿import seaborn as sns


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
