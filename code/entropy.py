import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def calculate_entropy(p, alpha):
    """
    Generalized Jensen-Shannon Divergence
    ref: https://arxiv.org/abs/1706.08671
    """
    if not sp.issparse(p):
        p = sp.csr_matrix(p.ravel())

    if alpha == 0:
        H = p.shape[1] - 1
    elif alpha == 1:
        H = - np.sum(p.data * np.log(p.data))
    elif alpha == 2:
        H = 1 - (p.data ** 2).sum()
    else:
        H = ((p.data ** alpha).sum() - 1)/ (1 - alpha)
    return H


def calculate_max_entropy(h1, h2, pi1=0.5, pi2=0.5, alpha=2):
    """
    Maximum entropy
    h1 : entropy of probability distribution p1
    h2 : entropy of probability distribution p2
    """
    if alpha == 1:
        d_max = - pi1  * np.log(pi1) - pi2 * np.log(pi2)
    else:
        d_max = (pi1 ** alpha - pi1) * h1 + \
            (pi2 ** alpha - pi2) * h2 + \
            (pi1 ** alpha + pi2 ** alpha - 1)/(1 - alpha)
    return d_max


def distance_language(p1, p2, alpha=2, norm=True):
    """
    Distance between two articles using Jensen-Shannon Divergence
    with given alpha parameter
    """
    h1 = calculate_entropy(p1, alpha)
    h2 = calculate_entropy(p2, alpha)
    h12 = calculate_entropy(0.5 * p1 + 0.5 * p2, alpha=alpha)
    d_lang = h12 - (0.5 * h1) - (0.5 * h2)
    if norm:
        d_max = calculate_max_entropy(h1, h2, pi1=0.5, pi2=0.5, alpha=alpha)
        d_lang = d_lang / d_max
    return d_lang


def distance_language_cosine(p1, p2):
    """
    Distance between two articles using cosine distance
    """
    d_lang = (1 - cosine_similarity(p1, p2)).ravel()[0]
    return d_lang


def calculate_hamming_dist(topic_text_1, topic_text_2):
    """
    Calculate Hamming distance between two topics text e.g.

    Examples
    ========
    calculate_hamming_dist('D.01', 'F.02') >> 2
    calculate_hamming_dist('F.01', 'F.02') >> 1
    calculate_hamming_dist('F.02', 'F.02') >> 0
    """
    topic_text_1 = topic_text_1.replace('Stroke', '').replace('Tauopathies,', '')
    topic_text_2 = topic_text_2.replace('Stroke', '').replace('Tauopathies,', '')
    if topic_text_1[-1] == '.':
        topic_text_1 = topic_text_1[:-1]
    if topic_text_2[-1] == '.':
        topic_text_2 = topic_text_2[:-1]
    topic1, subtopic1 = topic_text_1.split('.')
    topic2, subtopic2 = topic_text_2.split('.')
    if topic1 == topic2 and subtopic1 == subtopic2:
        return 0
    elif topic1 == topic2 and subtopic1 != subtopic2:
        return 1
    else:
        return 2
