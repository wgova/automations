import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, DBSCAN
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import fcluster, ward, dendrogram
from scipy.cluster.vq import kmeans, vq
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA

# Functions for clustering
# given a linkage model, plot dendogram, with the colors indicated by the a cutoff point at which we define clusters
# Example from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
def plot_dendrogram(raw_ts_dataframe, name_of_dataset, **kwargs):
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0).fit(
        raw_ts_dataframe.T.values
    )
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    # Create linkage matrix
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    # Plot the corresponding dendrogram
    fig = plt.figure(figsize=(10, 5.5))
    fig.suptitle(f"Hierarchical clusters: {name_of_dataset}")
    dendrogram(linkage_matrix, **kwargs)
    plt.savefig(f"hierarchical_{name_of_dataset}")
    return linkage_matrix


def plot_elbow_silhoutte_k_evaluation(name_of_data: str, data_array, max_clusters):
    range_n_clusters = range(2, max_clusters)
    elbow = []
    s_score = []
    for n_clusters in range_n_clusters:
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            init="k-means++",
            max_iter=1000,
            n_init=1,
        )
        cluster_labels = clusterer.fit_predict(data_array)
        # Average silhouette score
        silhouette_avg = silhouette_score(data_array, cluster_labels)
        s_score.append(silhouette_avg)
        # Average SSE"
        elbow.append(
            clusterer.inertia_
        )  # Inertia: Sum of distances of samples to their closest cluster center
    fig = plt.figure(figsize=(15, 5.5))
    fig.suptitle(f"K-means clusters for {name_of_data}", fontsize=16)
    fig.add_subplot(121)
    plt.plot(range_n_clusters, elbow, "g-", label=f"{name_of_data} SSE")
    plt.xlabel("Cluster")
    plt.ylabel("Sum of squared error(SSE) / Distortion")
    
    fig.add_subplot(122)
    fig.suptitle("Silhouette method results")
    plt.plot(range_n_clusters, s_score, "b-", label=f"{name_of_data} \n Silhouette Score")
    plt.xlabel("Cluster")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.show()

def plot_kmeans_clusters(data_array,number_of_clusters,name_of_data:str):
  # computing K-Means with K = number_of_clusters
  centroids,_ = kmeans(data_array,number_of_clusters,random_state=42,thresh=1e-05,n_init=1)
  # assign each sample to a cluster
  idx,_ = vq(data_array,centroids)
  # some plotting using numpy's logical indexing
  fig, ax = plt.subplots(figsize=(10,5.5))
  fig.suptitle(f"K-means clusters for {name_of_data}", fontsize=12)
  for cluster in range(number_of_clusters):
    colours = ['ob','ok','or','og','om','oc','oy']
    ax.plot(data_array[idx==cluster,0],data_array[idx==cluster,1],colours[cluster],label=f'cluster {cluster}')
    plt.legend()
  plt.savefig(f"{PATH}/images/k_means_{name_of_data}")
  plt.show()
  return idx

def get_clustered_features(product_name,df_features,experiment,PATH):
  #product_name = 'all_products'
  # Reduce dimensions using PCA
  pca = PCA(n_components=2)
  principalComponents = pca.fit_transform(df_features)
  # Save components to a DataFrame
  PCA_components = pd.DataFrame(principalComponents)
  # Plot the explained variances
  features = range(pca.n_components_)

  # Optimum clusters
  # plot_elbow_silhoutte_k_evaluation(f"{product_name}_{experiment}_pca_kmeans",np.asarray(PCA_components),15)
  kelbow_visualizer = KElbowVisualizer(
      KMeans(random_state=42), k=15,metric='distortion',
      timings=False,locate_elbow=True,size=(512, 340))
  kelbow_visualizer.fit(np.asarray(PCA_components))
  pca_k_value = kelbow_visualizer.elbow_value_
  plt.title('Locating optimum number of clusters (k) using the elbow method')
  plt.legend()
  #plt.savefig(f"{PATH}/images/{experiment}_elbow")

  clusters_features_uncorrelated = plot_kmeans_clusters(np.asarray(PCA_components),pca_k_value,f"{product_name}_{experiment}_pca_kmeans",f"{PATH}/images")

  details = [(name,cluster) for name, cluster in zip(df_features.index,clusters_features_uncorrelated)]
  cluster_df = pd.DataFrame(details,columns=['names','cluster'])
  cluster_df['names'].astype('category')
  get_names = df_features.reset_index().rename(columns={'country_product':'names'})
  get_names.names.astype('category')
  country_cluster = pd.merge(get_names,cluster_df,how='inner', on='names')
  groups = country_cluster.groupby(['cluster']).agg('mean')

  dict_clust = {0:'cluster_0',
                1: 'cluster_1',
                2: 'cluster_2',
                3: 'cluster_3',
                4: 'cluster_4',
                5: 'cluster_5'
                }
  clust = groups.reset_index()
  clust.replace({'cluster': dict_clust},inplace=True)
  clust.set_index('cluster',inplace=True)
  # x = clust.iloc[-1,:]
  cluster_features = clust.T
  
  n = len(cluster_features.columns)
  fig, ax = plt.subplots(n, 1, figsize=(10, n * 3), sharex=True,sharey=True)
  for i in range(n):
      plt.sca(ax[i])
      col = cluster_features.columns[i]
      cluster_features[col].plot(kind='bar')
      plt.title(f"Features for {col}")
      plt.tight_layout()
  #fig.savefig(f"{PATH}/images/{product_name}_{experiment}_pca_kmeans_features.png",bbox_inches = "tight")
  return country_cluster,cluster_features
