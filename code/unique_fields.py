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

from sklearn.externals import joblib
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
