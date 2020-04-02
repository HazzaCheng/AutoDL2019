from models.speech.neural_model.attention_gru import AttentionGruModel
from models.speech.neural_model.bilstm_attention import BilstmAttentionModel
from models.speech.neural_model.cnn import Cnn2DModel
from models.speech.neural_model.crnn2d import Crnn2dModel
from models.speech.neural_model.lstm_attention import LstmAttentionModel
from models.speech.simple_model.logistic_regression import LogisticRegression
from models.speech.neural_model.resnet.model import Model as ThinResNet34

# MODEL NAME
CNN2D_MODEL = 'cnn2d'
CRNN2D_MODEL = 'crnn2d'
SVM_MODEL = 'svm'
BILSTM_MODEL = 'bilstm'
LSTM_MODEL = 'lstm'
LR_MODEL = 'lr'
ATT_GRU_MODEL = 'att_gru'
THIN_RESNET34_MODEL = 'thin_resnet34'

# MODEL LIB
SPEECH_MODEL_LIB = {
    LR_MODEL: LogisticRegression,
    LSTM_MODEL: LstmAttentionModel,
    CRNN2D_MODEL: Crnn2dModel,
    BILSTM_MODEL: BilstmAttentionModel,
    CNN2D_MODEL: Cnn2DModel,
    ATT_GRU_MODEL: AttentionGruModel,
    THIN_RESNET34_MODEL: ThinResNet34
}
