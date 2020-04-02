from features.speech.speech_features import MFCC, MEL_SPECTROGRAM
from models.nlp import NLP_CNN, NLP_GRU
from models.speech import LR_MODEL, LSTM_MODEL, THIN_RESNET34_MODEL

# constants
CLASS_NUM = 'class_num'

MAX_VALID_PER_CLASS = 100
MAX_VALID_SET_SIZE = 500
MIN_VALID_PER_CLASS = 1

# ----------------------------------- AutoSpeech ----------------------------------- #

# SAMPLE HYPER PARAMETERS
SPEECH_SIMPLE_MODEL_SAMPLE_MAX_RUN_LOOP = 3
SPEECH_NEURAL_MODEL_SAMPLE_MAX_RUN_LOOP = 8
SPEECH_SIMPLE_MODEL_MAX_SAMPLE_NUM = 4000
SPEECH_SIMPLE_MODEL_EACH_LOOP_SAMPLE_NUM = 500
SPEECH_NEURAL_MODEL_EACH_LOOP_SAMPLE_NUM = 200

# MODEL SEQUENCES
SPEECH_SIMPLE_MODEL = LR_MODEL
# SPEECH_SIMPLE_NEURAl_MODEL_SEQUENCES = [LSTM_MODEL, CRNN2D_MODEL, CNN2D_MODEL, BILSTM_MODEL, ATT_GRU_MODEL]
SPEECH_SIMPLE_NEURAl_MODEL_SEQUENCES = [LSTM_MODEL]
SPEECH_COMPLEX_NEURAl_MODEL = THIN_RESNET34_MODEL
SPEECH_ROUND0_MODEL_MAX_RUN_LOOP = 12
SPEECH_MODEL_MAX_RUN_LOOP = 100

# FEATURE
SPEECH_FEATURES_SEQUENCES = {'simple_model_feature': {'name': MEL_SPECTROGRAM, 'max_duration': 5,
                                                      'kwargs': {'n_mels': 30, 'use_power_db': True}},
                             'simple_neural_model_feature': [
                                 {'name': MFCC, 'max_duration': 10,
                                  'kwargs': {'n_mfcc': 96}},
                                 {'name': MEL_SPECTROGRAM, 'max_duration': 10,
                                  'kwargs': {'n_mels': 128}},
                                 {'name': MFCC, 'max_duration': 30,
                                  'kwargs': {'n_mfcc': 128}},
                                 {'name': MEL_SPECTROGRAM, 'max_duration': 30,
                                  'kwargs': {'n_mels': 128}},
                                 {'name': MFCC, 'max_duration': 50,
                                  'kwargs': {'n_mfcc': 128}},
                                 {'name': MEL_SPECTROGRAM, 'max_duration': 50,
                                  'kwargs': {'n_mels': 128}}]
                             }

# ----------------------------------- AutoNLP ----------------------------------- #
# EN_EMBEDDING_PATH = '/home/chengfeng/embedding/cc.en.300.vec.gz'
# ZH_EMBEDDING_PATH = '/home/chengfeng/embedding/cc.zh.300.vec.gz'
EN_EMBEDDING_PATH = '/app/embedding/cc.en.300.vec.gz'
ZH_EMBEDDING_PATH = '/app/embedding/cc.zh.300.vec.gz'
NLP_PER_CLASS_READ_NUM = 1000
NLP_MAX_VOCAB_SIZE = 10000
IS_LOAD_EMBEDDING = True
LOAD_EMBEDDING_DIM = 300
ZH = 'ZH'
EN = 'EN'
NLP_MAX_TRAIN_PER_CLASS = 500
NLP_TFIDF_TRAIN_NUM = 5000
NLP_READ_CUT_LEN = 300

NLP_FAST_MODEL_RUN_LOOP = 3
NLP_SIMPLE_NEURAl_MODEL_SEQUENCES = [NLP_CNN, NLP_GRU]
NLP_FEATURES_SEQUENCES = {
    'fast_model_feature':
        {'name': 'tfidf', 'level': 0, 'embedding_dim': None, 'is_pretrained_embedding': None, 'is_use_new_data': True},
    'simple_model_feature':
        {'name': 'char-level', 'level': 1, 'embedding_dim': 64, 'is_pretrained_embedding': None, 'is_use_new_data': True},
    'simple_neural_model_feature': [
        {'name': 'char-level', 'level': 1, 'embedding_dim': 64,
         'is_pretrained_embedding': None, 'is_use_new_data': True},
        {'name': 'word-level + pretrained embedding300dim', 'level': 2, 'embedding_dim': 300,
         'is_pretrained_embedding': True, 'is_use_new_data': True},
        {'name': 'word-level + embedding64dim', 'level': 2, 'embedding_dim': 64,
         'is_pretrained_embedding': False, 'is_use_new_data': False}
    ]
}
