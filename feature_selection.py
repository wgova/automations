import re

def remove_features_df(df, exclude_list):
    feats = df.columns
    features_filtered_df = df[feats[~feats.isin(exclude_list)]]
    return features_filtered_df


def select_features_df(df, features_list):
    feats_df = df[features_list]
    return feats_df


def filter_features(features_df, features_to_exclude):
    exclude_list = []
    for i in features_to_exclude:
        feature_cols = features_df.filter(regex=i, axis=1).columns.to_list()
        exclude_list.extend(feature_cols)
    return exclude_list
