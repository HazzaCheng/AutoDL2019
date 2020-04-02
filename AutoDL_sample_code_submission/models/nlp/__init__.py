#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-12-10

from .neural_model.text_cnn import TextCnn
from .neural_model.cnn import Cnn
from .neural_model.gru import Gru
from .neural_model.gru_attention import GruAttention
from .neural_model.tf_model import TFModel
from .simple_model.svm import Svm
from .simple_model.lr import Lr

# MODEL NAME
NLP_SVM = 'nlp_svm'
NLP_LR = 'nlp_lr'
NLP_TEXT_CNN = 'nlp_text_cnn'
NLP_CNN = 'nlp_cnn'
NLP_GRU = 'nlp_gru'
NLP_GRU_ATTENTION = 'nlp_gru_attention'
NLP_TF_MODEL = 'nlp_tf_model'

# MODEL LIB
NLP_MODEL_LIB = {
    NLP_SVM: Svm,
    NLP_LR: Lr,
    NLP_TEXT_CNN: TextCnn,
    NLP_CNN: Cnn,
    NLP_GRU: Gru,
    NLP_GRU_ATTENTION: GruAttention,
    NLP_TF_MODEL: TFModel
}