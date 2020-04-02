import logging
import sys

import torch
import torchvision.models as models
from torch.utils import model_zoo
from torchvision.models.squeezenet import model_urls

from ..import skeleton

formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s %(filename)s] %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(handler)


class SqueezeneNet11(models.SqueezeNet):


    def __init__(self, in_channels, num_classes=10):
        super(SqueezeneNet11, self).__init__(version=1.1, num_classes=num_classes)

        if in_channels == 3:
            self.stem = torch.nn.Sequential(
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
            )
        elif in_channels == 1:
            self.stem = torch.nn.Sequential(
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                skeleton.nn.CopyChannels(3),
            )
        else:
            self.stem = torch.nn.Sequential(
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                torch.nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False)
            )

        final_conv = torch.nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            final_conv,
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self._half = False
        self._class_normalize = True

    def init(self, model_dir, gain=1.):
        sd = model_zoo.load_url(model_urls['squeezenet1_1'], model_dir=model_dir)
        del sd['classifier.1.weight']
        del sd['classifier.1.bias']
        self.load_state_dict(sd, strict=False)

        for idx in range(len(self.stem)):
            m = self.stem[idx]
            if hasattr(m, 'weight'):
                torch.nn.init.xavier_normal_(m.weight, gain=gain)
                LOGGER.debug('initialize stem weight')
        torch.nn.init.xavier_uniform_(self.classifier[1].weight, gain=gain)
        LOGGER.debug('initialize classifier weight')

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):
        inputs = self.stem(inputs)
        logits = models.SqueezeNet.forward(self, inputs)
        logits /= tau

        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)

        if self._class_normalize and isinstance(self.loss_fn, (torch.nn.BCEWithLogitsLoss,
                                                               skeleton.nn.BinaryCrossEntropyLabelSmooth)):
            pos = (targets == 1).to(logits.dtype)
            neg = (targets < 1).to(logits.dtype)
            npos = pos.sum()
            nneg = neg.sum()

            positive_ratio = max(0.1, min(0.9, (npos) / (npos + nneg)))
            negative_ratio = max(0.1, min(0.9, (nneg) / (npos + nneg)))
            LOGGER.debug('[BCEWithLogitsLoss] positive_ratio:%f, negative_ratio:%f',
                         positive_ratio, negative_ratio)

            normalized_loss = (loss * pos) / positive_ratio
            normalized_loss += (loss * neg) / negative_ratio

            loss = normalized_loss

        if reduction == 'avg':
            loss = loss.mean()
        elif reduction == 'max':
            loss = loss.max()
        elif reduction == 'min':
            loss = loss.min()
        return logits, loss

    def half(self):
        for module in self.modules():
            if len([c for c in module.children()]) > 0:
                continue

            if not isinstance(module, torch.nn.BatchNorm2d):
                module.half()
            else:
                module.float()
        self._half = True
        return self
