# -*- coding: utf-8 -*-
"""Demo.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13EJ8jctB-1gQuw6eT2k_Z7bzQeF3akEp

# Mental Health Demo

The purpose of this code is to provide a working framework on how the ML Model developed for Mental Health Analysis could be used.

## Setup environment
"""

import pandas as pd
import numpy as np
import nltk
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from nltk.tokenize import word_tokenize

! pip install gensim

! pip install fasttext

# Importing Fasttext
import fasttext

# Importing nltk configuration
nltk.download('punkt_tab')

from google.colab import drive

drive.mount('/content/data')

! wget 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz'

! gunzip 'cc.en.300.bin.gz'

# Loading the fasttext model created during the EDA
ft_model = fasttext.load_model('cc.en.300.bin')

# Loading the gradient boost model created during the model analysis
gb_model = joblib.load('/content/data/MyDrive/Colab Notebooks/model.joblib')

# Disorder Labels identified during the EDA
disorder_labels = ['anxiety', 'bipolar', 'depression', 'normal','personality_disorder', 'stress', 'suicidal']

# Get a list of stopwords to extract from the corpous
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
stopwords_list = list(ENGLISH_STOP_WORDS)

"""## Functions"""

def sentence_vector(tokens, model):
  '''
    Using a tokenized sentes provided by tokens, and the fasttext model (model)
    return a vector for the sentence.
  '''
  vectors = [model[word] for word in tokens if word not in stopwords_list and word in model]
  return np.mean(vectors, axis=0) if vectors else np.zeros(model.get_dimension())

def classify_prediction(prediction):
  '''
    This simply returns the value of the predicted model
  '''
  return disorder_labels[prediction]

"""## Demo"""

corpus = [
    'I have headaches all the time.',
    'I think she is looking at me',
    'Why cant\'t I catch a break on this job. ',
    'Don\'t touch me, you hear me.',
    'I always feel like someone is looking over my shoulder at work.',
    'oh my gosh!',
    'I\'m really worried, I want to cry.',
    'It\'s all my fault. It\'s always my fault.',
    "I guess I'm just burned out of people. A lot of bad things happened in my relational and social life since childhood till last year and I'm really bitter about it right now. Of course I want to find real connection, where I could be myself openly and comfortably, but because it always goes wrong and brings disappointments I'm really hopeless at this point and honestly kinda angry. I don't have patience for people anymore, I'm tired, I'm being judgy as hell and I'm activelly avoiding new people."
]

# finally let's unzip that zip file
! unzip '/content/data/MyDrive/Colab Notebooks/sentiment-analysis-for-mental-health.zip'

# The file has its own index so we will set the first column to the index value.
mh_df = pd.read_csv('/content/Combined Data.csv', index_col=0)

# Override corpus with content from the dataset.
# corpus = mh_df['statement'].to_list()

corpus[:10]

from math import isnan
feature_names = [ str(i) for i in range(0, 300)]
count = 0

for s in corpus:
  if type(s) != str:
    continue
  # print(count, s)
  count += 1
  tokens = word_tokenize(s)
  tokens_list = [ word.lower() for word in tokens ]
  vector = sentence_vector(tokens, ft_model)

  vector_df = pd.DataFrame(data=[vector], columns=feature_names)

  result = gb_model.predict(vector_df)
  # if result[0] != 3:
  print(s, ' ', classify_prediction(result[0]))

  if count > 21:
    break
