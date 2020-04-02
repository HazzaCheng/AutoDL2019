import logging
import sys
from collections import OrderedDict

import torch
from torch.utils import model_zoo
from .mobilenet_orig import model_urls, MobileNetV2
from ..skeleton.utils.tools import *
from ..import skeleton

formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s %(filename)s] %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(handler)


class MobileNetv2(MobileNetV2):

    def __init__(self, in_channels, num_classes=10, **kwargs):
        super(MobileNetv2, self).__init__(num_classes=num_classes, **kwargs)

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

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.last_channel, num_classes),
        )


        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = torch.nn.Linear(self.last_channels, num_classes, bias=False)
        self._half = False
        self._class_normalize = True
        self._is_video = False

    def set_video(self, is_video=True, times=False):
        self._is_video = is_video
        if is_video:
            self.conv1d_prev = torch.nn.Sequential(
                skeleton.nn.SplitTime(times),
                skeleton.nn.Permute(0, 2, 1, 3, 4),
            )

            self.conv1d_post = torch.nn.Sequential(
            )

    def is_video(self):
        return self._is_video

    def init(self, model_dir, gain=1.):
        # sd = model_zoo.load_url(model_urls['mobilenet_v2'], model_dir=model_dir).half()
        sd = model_zoo.load_url(model_urls['mobilenet_v2'], model_dir=model_dir)
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

    def forward_origin(self, x):
        x = MobileNetV2.forward(self, x)
        x = self.pool(x)
        if self.is_video():
            x = self.conv1d_prev(x)
            # log("x_conv1d_prev_size: {} {} {} {} {}".format(x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)))
            x = x.view(x.size(0), x.size(1), -1)
            # log("x_view_size: {} {} {}".format(x.size(0), x.size(1), x.size(2)))
            x = self.conv1d(x)
            # log("x_conv1d_size: {} {}".format(x.size(0), x.size(1)))
            # x = self.conv1d_post(x)
        x = x.view(x.size(0), -1)
        x = MobileNetv2.forward_linear(self, x)
        return x

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):  # pylint: disable=arguments-differ
        dims = len(inputs.shape)

        if self.is_video() and dims == 5:
            batch, times, channels, height, width = inputs.shape
            inputs = inputs.view(batch * times, channels, height, width)

        inputs = self.stem(inputs)
        # logits = MobileNetV2.forward(self, inputs)
        if self.is_video():
            logits = self.forward_origin(inputs)
        else:
            logits = MobileNetV2.forward_image(self, inputs)
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
