# distutils: language = c++
# -*- coding: utf-8 -*-
# @Time:    2020/2/29 20:24
# @Author:  Mecthew

from cython cimport boundscheck, wraparound
import math
import numpy as np


@boundscheck(False)
@wraparound(False)
def build_tfidf(str[:] data,int num_sentence):
    cdef:
        int min_df = 1
        dict word2cnt = {}
        dict word2index = {}
        dict word2idf = {}
        dict word2docs = {}
        dict word_in_docs = {}
        int i = 0
        int ind = 0

    for i in range(num_sentence):
        line = data[i]
        word_in_docs
        for w in line:
            if w in word2cnt:
                word2cnt[w] += 1
            else:
                word2cnt[w] = 1
            if w in word2docs:
                word2docs[w].add(i)
            else:
                word2docs[w] = set()
                word2docs[w].add(i)

    for k,v in word2cnt.items():
        if v >= min_df:
            word2index[k] = ind
            ind += 1
            doc_set = word2docs[k]
            word2idf[k] = math.log(num_sentence/float(len(doc_set)))
    max_vocab_size = ind

    return word2index, word2idf, max_vocab_size


def tfidf_vectorize(str[:] data, int num_sentence, word2index, word2idf, int max_vocab_size):
    ans = np.zeros((num_sentence, max_vocab_size), dtype=np.float32)
    cdef:
        float[:,:] x_train = ans
        dict doc2word = {}

    for i in range(num_sentence):
        line = data[i]
        doc2word = {}
        for w in line:
            if w in doc2word:
                doc2word[w] += 1
            else:
                doc2word[w] = 1

        for k,v in doc2word.items():
            if k in word2index:
                x_train[i][word2index[k]] = v * word2idf[k] / len(line)

    return ans

