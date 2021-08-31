# coding: utf-8

import warnings
import numpy as np
import pandas as pd
from packaging import version
from pyclustering.cluster.clarans import clarans
from pyclustering.utils import timedcall
from sklearn.metrics import pairwise_distances
from pyclustering.utils import draw_clusters,average_inter_cluster_distance,average_intra_cluster_distance,average_neighbor_distance
import sklearn
from sklearn.metrics import (
    davies_bouldin_score, silhouette_score, pairwise_distances,calinski_harabasz_score
)

# They changed the name of calinski_harabaz_score in later version of sklearn:
# https://github.com/scikit-learn/scikit-learn/blob/c4733f4895c1becdf587b38970f6f7066656e3f9/doc/whats_new/v0.20.rst#id2012
sklearn_version = version.parse(sklearn.__version__)
nm_chg_ver = version.parse("0.23")
if sklearn_version >= nm_chg_ver:
    from sklearn.metrics import calinski_harabasz_score as _cal_score
else:
    from sklearn.metrics import calinski_harabaz_score as _cal_score


def _get_clust_pairs(clusters):
    return [(i, j) for i in clusters for j in clusters if i > j]


def _dunn(data=None, dist=None, labels=None):
    clusters = set(labels)
    inter_dists = [
        dist[np.ix_(labels == i, labels == j)].min()
        for i, j in _get_clust_pairs(clusters)
    ]
    intra_dists = [
        dist[np.ix_(labels == i, labels == i)].max()
        for i in clusters
    ]
    return min(inter_dists) / max(intra_dists)

def intra_distances(data=None, dist=None, labels=None):
    clusters = set(labels)
    intra_dists = [
        dist[np.ix_(labels == i, labels == i)]
        for i in clusters]
    print(intra_dists)
    return sum(intra_dists)/len(intra_dists)

def inter_distances(data=None, dist=None, labels=None):
    clusters = set(labels)
    inter_dists = [dist[np.ix_(labels == i, labels == j)]
        for i, j in _get_clust_pairs(clusters)]
    return sum(inter_dists)/len(inter_dists)

def dunn(dist, labels):
    return _dunn(data=None, dist=dist, labels=labels)


def cop(data, dist, labels):
    clusters = set(labels)
    cpairs = _get_clust_pairs(clusters)
    prox_lst = [
        dist[np.ix_(labels == i[0], labels == i[1])].max()
        for i in cpairs
    ]

    out_l = []
    for c in clusters:
        c_data = data[labels == c]
        c_center = c_data.mean(axis=0, keepdims=True)
        c_intra = pairwise_distances(c_data, c_center).mean()

        c_prox = [prox for pair, prox in zip(cpairs, prox_lst) if c in pair]
        c_inter = min(c_prox)

        to_add = len(c_data) * c_intra / c_inter
        out_l.append(to_add)

    return sum(out_l) / len(labels)


def _silhouette_score2(data=None, dist=None, labels=None):
    return silhouette_score(dist, labels, metric='precomputed')


def _davies_bouldin_score2(data=None, dist=None, labels=None):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'divide by zero')
        return davies_bouldin_score(data, labels)


def _calinski_harabaz_score2(data=None, dist=None, labels=None):
    return _cal_score(data, labels)

def clarans_labels(clarans_object):
        labels_clarans = clarans_object.get_clusters()
        labels=pd.DataFrame(labels_clarans).T.melt(var_name='clusters')\
            .dropna()
        labels['value']=labels.value.astype(int)
        labels=labels.sort_values(['value'])\
                .set_index('value')\
                    .values\
                        .flatten()
        return labels

def calculate_clarans_cvi(data,initial_cluster,dist=None):
        cvi_df = pd.DataFrame(columns=['avg_inter_dist','silhouette','calinski',
        'avg_intra_dist','davies','dunn'])
        df_list = data.values.tolist()
        dist=pairwise_distances(data)
        np.fill_diagonal(dist, 0)
        for k in range(initial_cluster,10):
            print(k)
            clarans_model = clarans(df_list,k,3,5)
            (_, result) =timedcall(clarans_model.process)
            labels =  clarans_labels(result)
            clusters = set(labels)
            intra_dists = [dist[np.ix_(labels == i, labels == i)].max() for i in clusters]
            inter_dists = [dist[np.ix_(labels == i, labels == j)].min() for i, j in _get_clust_pairs(clusters)]
            avg_intra_dist = sum(intra_dists)/len(intra_dists)
            avg_inter_dist =  sum(inter_dists)/len(inter_dists)
            sihlouette = silhouette_score(dist, labels, metric='precomputed')
            calinski = calinski_harabasz_score(data, labels)
            davies = davies_bouldin_score(data, labels)
            dunn_ = dunn(dist,labels)
            cvi_df.loc[k] = [avg_inter_dist,
            sihlouette,calinski,
            avg_intra_dist,
            davies,dunn_]
            print(cvi_df)
            del clarans_model
        return cvi_df