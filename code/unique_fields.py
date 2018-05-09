# script to generate figure 2

import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr, pearsonr
from scipy.stats import kstest, ks_2samp
from itertools import combinations, combinations_with_replacement
from entropy import distance_language, distance_language_cosine, calculate_hamming_dist
from langauge_dist_corr import average_topic_topic_dist

from sklearn.externals import joblib


# load model with fixed vocab 20,000 words
count_vec_model = joblib.load(os.path.join('pkl', 'count_vec_model.pkl'))

def shuffle_topic_within_year(df):
    """
    Shuffle topic within year
    """
    df_list = []
    for year, df_year in df.groupby('year'):
        partition_shuffle = df_year.topic.as_matrix()
        np.random.shuffle(partition_shuffle)
        df_year['topic'] = partition_shuffle
        df_list.append(sfn_year_df)
    df_shuffle = pd.concat(df_list)
    return df_shuffle


if __name__ == '__main__':

    n_iter = 200
    n_sample = 1
    df = pd.read_csv(os.path.join('..', 'data', 'sfn_abstracts.csv'))
    df_shuffle = shuffle_topic_within_year(df)

    # within year language variability
    d_neighbor, d_neighbor_shuffle = [], []

    print("Distance to neighboring fields")
    for year, df_year in df.groupby('year'):
        partition_map = {v: k for k, v in enumerate(sorted(df_year['topic_desc'].unique()))}
        partitions = list(partition_map.keys())
        D_lang = average_topic_topic_dist(df_year,
                                          n_iter=n_iter,
                                          n_sample=n_sample)
        n_topics = len(D_lang)
        d_neighbor.append([(year, partitions[i], np.sort(D_fields[i, :])[0],
                            np.sort(D_fields[i, :])[1]]) for i in range(n_topics)])
    d_neighbor_df = pd.DataFrame(d_neighbor,
                                 columns=['year', 'topic_desc', 'd_field', 'd_nearest_neighbor'])
    d_neighbor_df.to_csv(os.path.join('..', 'data', 'language_distance.csv'), index=False)

    print("Distance to neighboring fields, shuffle control")
    for year, df_year in df_shuffle.groupby('year'):
        partition_map = {v: k for k, v in enumerate(sorted(df_year['topic_desc'].unique()))}
        partitions = list(partition_map.keys())
        D_lang = average_topic_topic_dist(df_year,
                                          n_iter=n_iter,
                                          n_sample=n_sample)
        n_topics = len(D_lang)
        d_neighbor_shuffle.append([(year, partitions[i], np.sort(D_lang[i, :])[0],
                                    np.sort(D_lang[i, :])[1]]) for i in range(n_topics)])
    d_neighbor_shuffle_df = pd.DataFrame(d_neighbor_shuffle,
                                         columns=['year', 'topic_desc', 'd_field', 'd_nearest_neighbor'])
    d_neighbor_shuffle_df.to_csv(os.path.join('..', 'data', 'language_distance_shuffle.csv'), index=False)

    n_papers_df = pd.read_csv(os.path.join('..', 'data', 'n_papers_df.csv'))
