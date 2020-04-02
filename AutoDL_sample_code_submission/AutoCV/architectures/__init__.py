from .resnet import ResNet18, ResNet34
from .mobilenet import MobileNetv2
from .efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2
from .densenet import Densenet121, Densenet201
from .se_resnext import SeResNext50, SeResNext101

# MODEL NAME
RESNET18_MODEL = 'resnet18'
RESNET34_MODEL = 'resnet34'
MOBILENET_MODEL = 'mobilenet'
EFFICIENT_MODEL = 'efficientnet'
DENSENET121_MODEL = 'densenet121'
DENSENET201_MODEL = 'densenet201'
SERESNEXT50_MODEL = 'se_resnext50'
SERESNEXT101_MODEL = 'se_resnext101'

# MODEL LIB
CV_MODEL_LIB = {
    RESNET18_MODEL: ResNet18,
    RESNET34_MODEL: ResNet34,
    MOBILENET_MODEL: MobileNetv2,
    EFFICIENT_MODEL: EfficientNetB0,
    DENSENET121_MODEL: Densenet121,
    DENSENET201_MODEL: Densenet201,
    SERESNEXT50_MODEL: SeResNext50,
    SERESNEXT101_MODEL: SeResNext101
}