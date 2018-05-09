# script to generate figure 3

import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr, pearsonr
from scipy.stats import kstest, ks_2samp
from itertools import combinations
from entropy import distance_language, distance_language_cosine

from sklearn.externals import joblib
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer

# load model with fixed vocab 20,000 words
count_vec_model = joblib.load(os.path.join('pkl', 'count_vec_model.pkl'))
sns.set(style="ticks")


def average_language_distance(df, n_iter=1000, n_sample=1, distance='cosine'):
    """
    Average language variability within topic

    Parameters
    ==========
    n_iter: number of iteration
    n_sample: sample size to compare abstracts between topic
    distance: 'cosine' or 'entropy'
        'cosine' will perform cosine similarity between abstracts
        'entropy' will perform Generalized Jensen-Shannon Divergence
    """
    distance_collected = []
    for topic, df_topic in df.groupby('topic'):
        for year, df_year in df_topic.groupby('year'):
            for i in range(n_iter):
                if i % 300 == 0:
                    sys.stdout.write("\ryear = %i, topic = %i, iteration = %i" % (year, topic, i))
                    sys.stdout.flush()
                ls = [' '.join(list(df_year['abstract_lemmatized'].sample(n=n_sample))),
                      ' '.join(list(df_year['abstract_lemmatized'].sample(n=n_sample)))]
                X = count_vec_model.transform(ls)
                P = normalize(X, axis=1, norm='l1')
                if distance == 'cosine':
                    d_lang = distance_language_cosine(P[0], P[1])
                else:
                    d_lang = distance_language(P[0], P[1], alpha=2) # entropy with alpha=2
                distance_collected.append([year, topic, d_lang])
    return pd.DataFrame(distance_collected,
                        columns=['year', 'topic', 'd_lang'])


def average_language_distance_yy(df, n_iter=5000, n_sample=1, distance='cosine'):
    """
    Average year-year language variability of a given topic

    Parameters
    ==========
    n_iter: number of iteration
    n_sample: sample size to compare abstracts between topic
    distance: 'cosine' or 'entropy'
        'cosine' will perform cosine similarity between abstracts
        'entropy' will perform Generalized Jensen-Shannon Divergence
    """
    distance_yy_collected = []
    for topic, df_topic in df.groupby('topic'):
        df_years = [(year, df_year) for year, df_year in df_topic.groupby('year')]
        for (year_1, df_year_1), (year_2, df_year_2) in combinations(df_years, 2):
            for i in range(n_iter):
                if i % 300 == 0:
                    sys.stdout.write("\ryear = %i, topic = %i, iteration = %i" % (year_1, topic, i))
                    sys.stdout.flush()
                ls = [' '.join(list(df_year_1['abstract_lemmatized'].sample(n=n_sample))),
                      ' '.join(list(df_year_2['abstract_lemmatized'].sample(n=n_sample)))]
                X = count_vec_model.transform(ls)
                P = normalize(X, axis=1, norm='l1')
                if distance == 'cosine':
                    d_lang = distance_language_cosine(P[0], P[1])
                else:
                    d_lang = distance_language(P[0], P[1], alpha=2)
                distance_yy_collected.append([year_1, year_2, topic, d_lang])
    return pd.DataFrame(distance_yy_collected,
                        columns=['year_prev', 'year', 'topic', 'd_lang'])


if __name__ == '__main__':

    df = pd.read_csv(os.path.join('..', 'data', 'sfn_abstracts.csv'))
    distance_yy_df = average_language_distance_yy(df, n_iter=1000, n_sample=1)
    distance_yy_df = distance_yy_df[(distance_yy_df['year'] - distance_yy_df['year_prev']) == 1]
    sort_distance_df = distance_yy_df.groupby('topic')[['d_lang']].mean().reset_index().sort_values('d_lang', ascending=False)
    topic_desc_df = pd.read_csv(os.path.join('..', 'data', 'topic_desc.csv'))
    partitions = list(sort_distance_df.topic)
    partitions_desc = list(sort_distance_df.merge(topic_desc_df).topic_description)

    plt.rcParams['figure.figsize'] = (20, 16)
    ax = sns.pointplot(x="topic", y="d_lang",
                       data=distance_yy_df,
                       ci=95, nboot=500,
                       order=partitions,
                       capsize=.0,
                       join=False,
                       color="tomato")
    plt.xlabel('Curated topic', fontsize=18)
    plt.ylabel('Average year-on-year language distance of a given topic', fontsize=20)
    plt.xticks(range(len(partitions_desc)), partitions_desc, rotation=90, fontsize=14)
    plt.yticks(fontsize=16)
    plt.ylim([0.85, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join('..', 'figures', 'figure_3.svg'))

    # size vs. distance
    n_papers_df = df.groupby(['year', 'topic'])[['subtopic']].count().reset_index().rename(columns={'subtopic': 'n_papers'})
    distance_average_df = distance_yy_df.groupby(['topic', 'year'])[['d_lang']].mean().reset_index()
    n_papers_df = distance_average_df.merge(n_papers_df, on=['topic', 'year']).query('n_papers >= 60')
    n_papers_df.to_csv(os.path.join('..', 'data', 'n_papers_df.csv'), index=False)
    print('Correlation between size and language distance = ', pearsonr(n_papers_df.d_lang, n_papers_df.n_papers))
