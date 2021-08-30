import re

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import (AgglomerativeClustering, 
KMeans,OPTICS, cluster_optics_dbscan)
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import clust_indices 
from clust_indices import (_dunn, cop, _davies_bouldin_score2, _silhouette_score2,_calinski_harabaz_score2,intra_distances,inter_distances)

class ValidClust:
    def __init__(self, k,
                 # No big deal that these are lists (i.e., mutable), given that
                 # we don't mutate them inside the class.
                 indices=['inter_dist','silhouette', 'davies','intra_dist','calinski','dunn'],
                 methods=['hierarchical', 'kmeans','kmedoids','dbscan','optics'],
                 linkage='ward', affinity='euclidean'):

        k, indices, methods = (
            [i] if type(i) in [int, str] else i
            for i in [k, indices, methods]
        )

        if linkage == 'ward' and affinity != 'euclidean':
            raise ValueError(
                "You must specify `affinity='euclidean'` when using the "
                "ward linkage type"
            )

        ok_indices = ['inter_dist','silhouette', 'davies','intra_dist','calinski','dunn', 'cop']
        ind_aliases = {i[0:3]: i for i in ok_indices}
        indices = [
            ind_aliases[i] if i in ind_aliases else i
            for i in indices
        ]
        for i in indices:
            if i not in ok_indices:
                raise ValueError('{0} is not a valid index value'.format(i))

        self.k = k
        self.indices = indices
        self.methods = methods
        self.linkage = linkage
        self.affinity = affinity

        self.score_df = None

    def __repr__(self):
        argspec = [
            '{}={}'.format('  ' + key, value)
            for key, value in self.__dict__.items() if key != 'score_df'
        ]
        argspec = ',\n'.join(argspec)
        argspec = re.sub('(linkage|affinity)=(\\w*)', "\\1='\\2'", argspec)
        return 'ValidClust(\n' + argspec + '\n)'

    def _get_method_objs(self):
        method_switcher = {
            'hierarchical': AgglomerativeClustering(),
            'kmeans': KMeans(random_state=42),
            'kmediods': KMedoids(),
            'dbscan' : DBSCAN(),
            'optics' : OPTICS()
        }
        objs = {i: method_switcher[i] for i in self.methods}
        for key, value in objs.items():
            if key == 'hierarchical':
                value.set_params(linkage=self.linkage, affinity=self.affinity)
            if key == 'dbscan':
                value.set_params(eps=0.15)
        return objs

    def _get_index_funs(self):
        index_fun_switcher = {
            'silhouette': _silhouette_score2,
            'davies': _davies_bouldin_score2,
            'calinski': _calinski_harabaz_score2,
            'dunn': _dunn,
            'intra_dist': intra_distances,
            'inter_dist': intra_distances,
            'cop': cop
        }
        return {i: index_fun_switcher[i] for i in self.indices}

    def fit(self, data):
        method_objs = self._get_method_objs()
        index_funs = self._get_index_funs()
        dist_inds = ['silhouette', 'dunn']

        d_overlap = [i for i in self.indices if i in dist_inds]
        if d_overlap:
            dist = pairwise_distances(data)
            np.fill_diagonal(dist, 0)
        else:
            dist = None

        index = pd.MultiIndex.from_product(
            [self.methods, self.indices],
            names=['method', 'index']
        )
        output_df = pd.DataFrame(
            index=index, columns=self.k, dtype=np.float64
        )

        for k in self.k:
          try:  
            for alg_name, alg_obj in method_objs.items():
                if alg_name == 'dbscan':
                  alg_obj.set_params(eps=(0.1*k), min_samples=10)
                elif alg_name =='optics':
                  alg_obj.set_params(min_samples=k)
                else:
                  alg_obj.set_params(n_clusters=k)
                labels = alg_obj.fit_predict(data)
                # have to iterate over self.indices here so that ordering of
                # validity indices is same in scores list as it is in output_df
                scores = [
                    index_funs[key](data, dist, labels)
                    for key in self.indices
                ]
                print('Clustering completed',alg_name,k,'clusters')
                output_df.loc[(alg_name, self.indices), k] = scores
          except ValueError:
            print('Failed',alg_name,k)
          finally:
            pass

        self.score_df = output_df
        return self

    def fit_predict(self, data):
        return self.fit(data).score_df

    def _normalize(self):
        score_df_norm = self.score_df.copy()
        for i in ['davies', 'cop']:
            if i in self.indices:
                score_df_norm.loc[(slice(None), i), :] = \
                    1 - score_df_norm.loc[(slice(None), i), :]
        normalize(score_df_norm, norm='max', copy=False)
        return score_df_norm

    def plot(self):
        norm_df = self._normalize()

        yticklabels = [',\n'.join(i) for i in norm_df.index.values]
        hmap = sns.heatmap(
            norm_df, cmap='Blues', cbar=False, yticklabels=yticklabels
        )
        hmap.set_xlabel('\nNumber of clusters')
        hmap.set_ylabel('Method, index\n')
        plt.tight_layout()
