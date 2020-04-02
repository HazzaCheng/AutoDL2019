#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-01-29
import gzip
import gc
import os
import numpy as np
import time

from tools import timeit, log
from Configurations import LOAD_EMBEDDING_DIM, NLP_MAX_VOCAB_SIZE, ZH, EN
from sklearn.feature_extraction.text import TfidfVectorizer
from features.nlp import ac
from features.nlp import tfidf
os.system('pip install jieba_fast')
NLP_FEATURE_DIR = os.path.dirname(__file__)


@timeit
def load_stopwords(stopwords_path):
    stopwords = []
    with open(stopwords_path, 'r', encoding='utf8') as fin:
        for line in fin:
            word = line.strip()
            if word:
                stopwords.append(word)
    return stopwords


@timeit
def load_embedding_dict(dict_path):
    embedding_dict = {}
    fin = gzip.open(dict_path, 'rb')
    for line in fin.readlines():
        values = line.strip().split()
        word = values[0].decode('utf-8')
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_dict[word] = coefs

    del fin, values, word, coefs
    gc.collect()

    return embedding_dict


@timeit
def get_embedding(embeddings_index, num_features, word_index):
    if embeddings_index is None:
        return None
    embedding_dim = LOAD_EMBEDDING_DIM
    embedding_matrix = np.zeros((num_features, embedding_dim))
    cnt = 0
    oov_words = []
    mean_vec = np.zeros(embedding_dim)
    for word, i in word_index.items():
        if i >= num_features:  # if index of word > num_features
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            mean_vec += embedding_vector
        else:
            # TODO: Words not found in embedding index should be the mean of all other word's embeddings.
            oov_words.append(i)
            # embedding_matrix[i] = np.zeros(embedding_dim)
            cnt += 1
    mean_vec = mean_vec / (num_features - cnt)
    for i in oov_words:
        embedding_matrix[i] = mean_vec

    print('fastText oov words: %s' % cnt)

    return embedding_matrix


@timeit
def vectorize_texts(lang, x, tfidf_vectorizer=None, stopwords=None, vocabulary=None):
    x = np.array(x, dtype='object')
    if lang == 'ZH':
        x = ac.clean_text_zh_seg1(x, None)
        x = np.array(x, dtype='object')
    elif lang == 'EN':
        x = ac.clean_text_en_seg1(x, None)
        x = [" ".join(text) for text in x]
        x = np.array(x, dtype='object')
    analyzer = 'char' if lang == ZH else 'word'
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=NLP_MAX_VOCAB_SIZE,
                                           vocabulary=vocabulary, analyzer=analyzer,
                                           stop_words=stopwords, decode_error='ignore')
        x_vectorized = tfidf_vectorizer.fit_transform(x)
    else:
        x_vectorized = tfidf_vectorizer.transform(x)
    return x_vectorized.toarray(), tfidf_vectorizer


@timeit
def tfidf_vectorize(lang, x, word2index=None, word2idf=None, max_vocab_size=None):
    t1 = time.time()
    x = np.array(x, dtype='object')
    if lang == 'ZH':
        x = ac.clean_text_zh_seg1(x, None)
    elif lang == 'EN':
        x = ac.clean_text_en_seg1(x, None)
    t2 = time.time()

    x = np.array(x, dtype='object')
    if word2index is None:
        word2index, word2idf, max_vocab_size = tfidf.build_tfidf(x, len(x))
    t3 = time.time()

    ret = tfidf.tfidf_vectorize(x, len(x), word2index, word2idf, max_vocab_size)
    t4 = time.time()
    log("clean {}s".format(t2-t1))
    log("build-tfidf {}s".format(t3-t2))
    log("vectorize-tfidf {}s".format(t4-t3))
    return ret, word2index, word2idf, max_vocab_size

