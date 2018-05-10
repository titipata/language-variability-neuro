# more script for model validation and figure 4s
import os
import pandas as pd
from scipy.stats import ks_2samp, pearsonr
import matplotlib.pyplot as plt


def test_linear_vs_mixture(df):
    """
    Test Bayesian Information Criteria between one and two parameters
    (We found that 1 parameter fits the data better using BIC)

    source: https://github.com/victorkristof/linear-regressions-mixture
    """
    from src import LinearRegressionsMixture

    X = np.atleast_2d(df.d_nearest_neighbor.as_matrix()).T
    y = np.atleast_2d(df.d_field.as_matrix()).T
    model_mixture = LinearRegressionsMixture(X, y, K=2)
    model_mixture.train(verbose=False)

    model = LinearRegressionsMixture(X, y, K=1)
    model.train(verbose=False)

    log_likelihood = - np.sum([(y_ - model.predict(x_)) ** 2 for x_, y_ in zip(X, y)])
    k = 1
    BIC = np.log(len(X)) * k - (2 * log_likelihood)

    log_likelihood_mixture = - np.sum([(y_ - model_mixture.predict(x_)) ** 2 for x_, y_ in zip(X, y)])
    k = 2
    BIC_mixture = np.log(len(X)) * k - (2 * log_likelihood_mixture)

    return BIC_mixture, BIC


def ks_test_distribution(df, df_shuffle):
    """
    Test to see that two sample of nearest neighbor of data and shuffle control
    are different
    """
    return ks_2samp(df.d_nearest_neighbor, df_shuffle.d_nearest_neighbor)


if __name__ == '__main__':

    lang_df = pd.read_csv(os.path.join('..', 'data', 'language_distance.csv'))
    lang_shuffle_df = pd.read_csv(os.path.join('..', 'data', 'language_distance_shuffle.csv'))
    n_papers_df = pd.read_csv(os.path.join('..', 'data', 'n_papers_df.csv'))[['topic', 'year', 'n_papers']]

    # correlation data, shuffle control
    print(pearsonr(lang_df.merge(n_papers_df).query("n_papers > 50").d_nearest_neighbor,
                   lang_df.merge(n_papers_df).query("n_papers > 50").d_field))
    print(pearsonr(lang_shuffle_df.merge(n_papers_df).query("n_papers > 50").d_nearest_neighbor,
                   lang_shuffle_df.merge(n_papers_df).query("n_papers > 50").d_field))
    # KS-test shows that distribution is different
    print(ks_test_distribution(lang_df, lang_shuffle_df))

    lang_var_df = pd.read_csv(os.path.join('..', 'data', 'language_var.csv'))
    lang_var_shuffle_df = pd.read_csv(os.path.join('..', 'data', 'language_var_shuffle.csv'))

    print(pearsonr(lang_var_df.query('n_papers > 50').d_nearest_neighbor,
                   lang_var_df.query('n_papers > 50').d_lang))
    print(pearsonr(lang_var_shuffle_df.query('n_papers > 50').d_nearest_neighbor,
                   lang_var_shuffle_df.query('n_papers > 50').d_lang))

    plt.subplot(1, 2, 1)
    plt.scatter(lang_var_shuffle_df.d_nearest_neighbor, lang_var_shuffle_df.d_lang,
                c='gray', alpha=0.4)
    plt.scatter(lang_var_df.d_nearest_neighbor, lang_var_df.d_lang,
                c='tomato', alpha=0.7)
    plt.xlabel('Language distance to nearest topic', fontsize=20)
    plt.ylabel('Year-on-year language distance', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(1, 2, 2)
    plt.scatter(lang_shuffle_df.d_nearest_neighbor, lang_shuffle_df.d_field,
                c='gray', alpha=0.4)
    plt.scatter(lang_df.d_nearest_neighbor, lang_df.d_field,
                c='tomato', alpha=0.7)
    plt.xlabel('Language distance to nearest topic', fontsize=20)
    plt.ylabel('Within year language distance', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join('..', 'figures', 'figure_4.svg'))
    plt.show()
