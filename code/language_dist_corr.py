# script to generate figure 2

import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr
from itertools import combinations_with_replacement
from entropy import distance_language, distance_language_cosine, calculate_hamming_dist

from sklearn.externals import joblib
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer

# load model with fixed vocab 20,000 words
count_vec_model = joblib.load(os.path.join('pkl', 'count_vec_model.pkl'))


def average_topic_topic_dist(df, n_iter=1000, n_sample=1, distance='cosine'):
    """
    Calculate language distance matrix between topics
    """
    partition_text = {topic: df_topic for topic, df_topic in df.groupby('topic_desc')}
    partition_map = {v: k for k, v in enumerate(sorted(df['topic_desc'].unique()))}
    n_topics = len(partition_map)
    D_fields_list = []
    for i in range(n_iter):
        if i % 5 == 0:
            sys.stdout.write("iteration = %i from %i iterations\r" % (i, n_iter))
            sys.stdout.flush()
        D_fields = np.zeros((n_topics, n_topics))
        for par1, par2 in combinations_with_replacement(partition_map.keys(), 2):
            df1 = partition_text[par1]
            df2 = partition_text[par2]
            ls = [' '.join(list(df1.abstract_lemmatized.sample(n=n_sample))),
                  ' '.join(list(df2.abstract_lemmatized.sample(n=n_sample)))]
            X = count_vec_model.transform(ls)
            P = normalize(X, axis=1, norm='l1')
            if distance == 'cosine':
                d_lang = distance_language_cosine(P[0], P[1])
            else:
                d_lang = distance_language(P[0], P[1], alpha=2)
            D_fields[partition_map[par1], partition_map[par2]] = d_lang
            D_fields[partition_map[par2], partition_map[par1]] = d_lang
        D_fields_list.append(D_fields)
    return np.mean(D_fields_list, axis=0)


def hamming_topic_topic_dist(df):
    """
    Calculate Hamming distance matrix between topics
    """
    partition_text = {topic: df_topic for topic, df_topic in df.groupby('topic_desc')}
    partition_map = {v: k for k, v in enumerate(sorted(df['topic_desc'].unique()))}
    n_topics = len(partition_map)
    D_expert = np.zeros((n_topics, n_topics))
    for par1, par2 in combinations_with_replacement(partition_map.keys(), 2):
        df1 = partition_text[par1]
        df2 = partition_text[par2]
        d_expert = calculate_hamming_dist(df1.iloc[0]['topic_desc'], df2.iloc[0]['topic_desc'])
        D_expert[partition_map[par1], partition_map[par2]] = d_expert
        D_expert[partition_map[par2], partition_map[par1]] = d_expert
    return D_expert


if __name__ == '__main__':

    n_iter = 100
    n_sample = 1
    df = pd.read_csv(os.path.join('..', 'data', 'sfn_abstracts.csv'))

    plt.rcParams['figure.figsize'] = (15, 15)
    corr_exp_lang = []
    for year, df_year in df.groupby('year'):
        D_lang = average_topic_topic_dist(df_year,
                                          n_iter=n_iter,
                                          n_sample=n_sample)
        D_expert = hamming_topic_topic_dist(df_year)
        n_topics = len(D_fields)
        corr_exp_lang.append([year,
                              pearsonr(D_expert[np.tril_indices(n_topics, k=0)],
                                       D_lang[np.tril_indices(n_topics, k=0)])])

        if year == 2017:
            # re-scale for visualization purpose
            D_expert = 2 - D_expert
            D_expert[np.triu_indices(len(D_expert), k=0)] = 2
            D_expert[np.diag_indices(len(D_expert))] = 1.6
            plt.matshow(D, cmap=cm.afmhot, interpolation='none', vmin=0, vmax=2)
            plt.title('Expert distance matrix')
            plt.axis('off')
            plt.savefig(os.path.join('..', 'figures', 'figure_2a.svg'))

            D_lang = np.mean(D_fields_list, axis=0)
            D_lang = 1 - (D_lang - np.min(D_lang)) / (np.max(D_lang) - np.min(D_lang))
            D_lang[np.triu_indices(len(D_lang), k=1)] = 1
            plt.matshow(D_fields)
            plt.title('Language distance matrix')
            plt.savefig(os.path.join('..', 'figures', 'figure_2b.svg'))

    corr_exp_lang = np.array([[y, r] for y, (r, p_value) in corr_exp_lang])
    plt.rcParams['figure.figsize'] = (7, 7)
    plt.scatter(corr_exp_lang[:, 0], corr_exp_lang[:, 1])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Correlation', fontsize=18)
    plt.title('Corr. between expert and langauge', fontsize=20)
    plt.xlim([2012, 2018])
    plt.yticks(np.arange(0.6, 0.75, 0.05))
    plt.tight_layout()
    plt.savefig(os.path.join('..', 'figures', 'figure_2c.svg'))
