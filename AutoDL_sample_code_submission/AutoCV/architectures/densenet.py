import logging
import sys
from collections import OrderedDict
import re

import torch
import torch.nn.functional as F
import torchvision.models as models
from torch.utils import model_zoo
from torchvision.models.densenet import model_urls, _DenseBlock, _Transition
from ..skeleton.utils.tools import *
from ..import skeleton

formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s %(filename)s] %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(handler)


class Densenet121(models.DenseNet):

    def __init__(self, in_channels, num_classes=10, **kwargs):
        num_init_features = 64
        growth_rate = 32
        block_config = (6, 12, 24, 16)
        bn_size = 4
        drop_rate=0

        super(Densenet121, self).__init__(num_init_features=num_init_features, growth_rate=growth_rate,
                                          block_config=block_config,
                                          num_classes=num_classes, **kwargs)

        if in_channels == 3:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
            )
        elif in_channels == 1:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                skeleton.nn.CopyChannels(3),
            )
        else:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                torch.nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(3),
            )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                num_features = num_features // 2

        self.last_channels = num_features

        self.conv1d = torch.nn.Sequential(
            skeleton.nn.Split(OrderedDict([
                ('skip', torch.nn.Sequential(
                    # torch.nn.AvgPool1d(3, stride=2, padding=1)
                )),
                ('deep', torch.nn.Sequential(
                    # torch.nn.Conv1d(self.last_channels, self.last_channels // 4,
                    #                  kernel_size=1, stride=1, padding=0, bias=False),
                    # torch.nn.BatchNorm1d(self.last_channels // 4),
                    # torch.nn.ReLU(inplace=True),
                    # torch.nn.Conv1d(self.last_channels // 4, self.last_channels // 4,
                    #                  kernel_size=5, stride=1, padding=2, groups=self.last_channels // 4, bias=False),
                    # torch.nn.BatchNorm1d(self.last_channels // 4),
                    # torch.nn.ReLU(inplace=True),
                    # torch.nn.Conv1d(self.last_channels // 4, self.last_channels,
                    #                  kernel_size=1, stride=1, padding=0, bias=False),
                    # torch.nn.BatchNorm1d(self.last_channels),

                    # torch.nn.Conv1d(self.last_channels, self.last_channels,
                    #                 kernel_size=3, stride=1, padding=1, bias=False),
                    #                 # kernel_size=5, stride=1, padding=2, bias=False),
                    # torch.nn.BatchNorm1d(self.last_channels),
                    # torch.nn.ReLU(inplace=True),
                    torch.nn.Conv1d(self.last_channels, self.last_channels,
                                    kernel_size=5, stride=1, padding=2, bias=False),
                    # kernel_size=5, stride=1, padding=2, bias=False),
                    torch.nn.BatchNorm1d(self.last_channels),
                    torch.nn.ReLU(inplace=True),
                ))
            ])),
            skeleton.nn.MergeSum(),

            # torch.nn.Conv1d(self.last_channels, self.last_channels,
            #                 kernel_size=5, stride=1, padding=2, bias=False),
            # torch.nn.BatchNorm1d(self.last_channels),
            # torch.nn.ReLU(inplace=True),

            torch.nn.AdaptiveAvgPool1d(1)
        )

        # self.self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # self.pool = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = torch.nn.Linear(self.last_channels, num_classes, bias=False)
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

    def init(self, model_dir=None, gain=1.):
        self.model_dir = model_dir if model_dir is not None else self.model_dir
        sd = model_zoo.load_url(model_urls['densenet121'], model_dir=self.model_dir)
        # sd = model_zoo.load_url(model_urls['resnet34'], model_dir='./models/')
        del sd['classifier.weight']
        del sd['classifier.bias']

        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(sd.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                sd[new_key] = sd[key]
                del sd[key]

        self.load_state_dict(sd, strict=False)

        # for idx in range(len(self.stem)):
        #     m = self.stem[idx]
        #     if hasattr(m, 'weight') and not isinstance(m, torch.nn.BatchNorm2d):
        #         # torch.nn.init.kaiming_normal_(self.stem.weight, mode='fan_in', nonlinearity='linear')
        #         torch.nn.init.xavier_normal_(m.weight, gain=gain)
        #         LOGGER.debug('initialize stem weight')
        #
        # for idx in range(len(self.conv1d)):
        #     m = self.conv1d[idx]
        #     if hasattr(m, 'weight') and not isinstance(m, torch.nn.BatchNorm1d):
        #         # torch.nn.init.kaiming_normal_(self.stem.weight, mode='fan_in', nonlinearity='linear')
        #         torch.nn.init.xavier_normal_(m.weight, gain=gain)
        #         LOGGER.debug('initialize conv1d weight')

        # torch.nn.init.kaiming_uniform_(self.fc.weight, mode='fan_in', nonlinearity='sigmoid')
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=gain)
        LOGGER.debug('initialize classifier weight')

    def forward_origin(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))

        if self.is_video():
            # con1d_prev: => (batch, channels, times, height, width)
            out = self.conv1d_prev(out)
            # => (batch, channels, times*height*width)
            out = out.view(out.size(0), out.size(1), -1)
            # conv1d: conv on last dim
            out = self.conv1d(out)
            out = self.conv1d_post(out)
        if self.is_video():
            out = out.view(out.size(0), -1)
        else:
            out = out.view(features.size(0), -1)

        out = self.classifier(out)
        return out

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):  # pylint: disable=arguments-differ
        dims = len(inputs.shape)

        if self.is_video() and dims == 5:
            batch, times, channels, height, width = inputs.shape
            inputs = inputs.view(batch*times, channels, height, width)

        inputs = self.stem(inputs)

        logits = self.forward_origin(inputs)

        logits /= tau

        if targets is None:
            return logits

        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)

        if self._class_normalize and isinstance(self.loss_fn, (torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth)):
            pos = (targets == 1).to(logits.dtype)
            neg = (targets < 1).to(logits.dtype)
            npos = pos.sum()
            nneg = neg.sum()

            positive_ratio = max(0.1, min(0.9, (npos) / (npos + nneg)))
            negative_ratio = max(0.1, min(0.9, (nneg) / (npos + nneg)))
            LOGGER.debug('[BCEWithLogitsLoss] positive_ratio:%f, negative_ratio:%f',
                         positive_ratio, negative_ratio)

            normalized_loss =  (loss * pos) / positive_ratio
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
        # super(BasicNet, self).half()
        for module in self.modules():
            if len([c for c in module.children()]) > 0:
                continue

            if not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.half()
            else:
                module.float()
        self._half = True
        return self


class Densenet201(models.DenseNet):

    def __init__(self, in_channels, num_classes=10, **kwargs):
        num_init_features = 64
        growth_rate = 32
        block_config = (6, 12, 48, 32)
        bn_size = 4
        drop_rate=0

        super(Densenet201, self).__init__(num_init_features=num_init_features, growth_rate=growth_rate,
                                          block_config=block_config,
                                          num_classes=num_classes, **kwargs)

        if in_channels == 3:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
            )
        elif in_channels == 1:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                skeleton.nn.CopyChannels(3),
            )
        else:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                torch.nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(3),
            )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                num_features = num_features // 2

        self.last_channels = num_features

        self.conv1d = torch.nn.Sequential(
            skeleton.nn.Split(OrderedDict([
                ('skip', torch.nn.Sequential(
                    # torch.nn.AvgPool1d(3, stride=2, padding=1)
                )),
                ('deep', torch.nn.Sequential(
                    # torch.nn.Conv1d(self.last_channels, self.last_channels // 4,
                    #                  kernel_size=1, stride=1, padding=0, bias=False),
                    # torch.nn.BatchNorm1d(self.last_channels // 4),
                    # torch.nn.ReLU(inplace=True),
                    # torch.nn.Conv1d(self.last_channels // 4, self.last_channels // 4,
                    #                  kernel_size=5, stride=1, padding=2, groups=self.last_channels // 4, bias=False),
                    # torch.nn.BatchNorm1d(self.last_channels // 4),
                    # torch.nn.ReLU(inplace=True),
                    # torch.nn.Conv1d(self.last_channels // 4, self.last_channels,
                    #                  kernel_size=1, stride=1, padding=0, bias=False),
                    # torch.nn.BatchNorm1d(self.last_channels),

                    # torch.nn.Conv1d(self.last_channels, self.last_channels,
                    #                 kernel_size=3, stride=1, padding=1, bias=False),
                    #                 # kernel_size=5, stride=1, padding=2, bias=False),
                    # torch.nn.BatchNorm1d(self.last_channels),
                    # torch.nn.ReLU(inplace=True),
                    torch.nn.Conv1d(self.last_channels, self.last_channels,
                                    kernel_size=5, stride=1, padding=2, bias=False),
                    # kernel_size=5, stride=1, padding=2, bias=False),
                    torch.nn.BatchNorm1d(self.last_channels),
                    torch.nn.ReLU(inplace=True),
                ))
            ])),
            skeleton.nn.MergeSum(),

            # torch.nn.Conv1d(self.last_channels, self.last_channels,
            #                 kernel_size=5, stride=1, padding=2, bias=False),
            # torch.nn.BatchNorm1d(self.last_channels),
            # torch.nn.ReLU(inplace=True),

            torch.nn.AdaptiveAvgPool1d(1)
        )

        # self.self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # self.pool = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = torch.nn.Linear(self.last_channels, num_classes, bias=False)
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

    def init(self, model_dir=None, gain=1.):
        self.model_dir = model_dir if model_dir is not None else self.model_dir
        sd = model_zoo.load_url(model_urls['densenet201'], model_dir=self.model_dir)
        # sd = model_zoo.load_url(model_urls['resnet34'], model_dir='./models/')
        del sd['classifier.weight']
        del sd['classifier.bias']

        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(sd.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                sd[new_key] = sd[key]
                del sd[key]

        self.load_state_dict(sd, strict=False)

        # for idx in range(len(self.stem)):
        #     m = self.stem[idx]
        #     if hasattr(m, 'weight') and not isinstance(m, torch.nn.BatchNorm2d):
        #         # torch.nn.init.kaiming_normal_(self.stem.weight, mode='fan_in', nonlinearity='linear')
        #         torch.nn.init.xavier_normal_(m.weight, gain=gain)
        #         LOGGER.debug('initialize stem weight')
        #
        # for idx in range(len(self.conv1d)):
        #     m = self.conv1d[idx]
        #     if hasattr(m, 'weight') and not isinstance(m, torch.nn.BatchNorm1d):
        #         # torch.nn.init.kaiming_normal_(self.stem.weight, mode='fan_in', nonlinearity='linear')
        #         torch.nn.init.xavier_normal_(m.weight, gain=gain)
        #         LOGGER.debug('initialize conv1d weight')

        # torch.nn.init.kaiming_uniform_(self.fc.weight, mode='fan_in', nonlinearity='sigmoid')
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=gain)
        LOGGER.debug('initialize classifier weight')

    def forward_origin(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))

        if self.is_video():
            # con1d_prev: => (batch, channels, times, height, width)
            out = self.conv1d_prev(out)
            # => (batch, channels, times*height*width)
            out = out.view(out.size(0), out.size(1), -1)
            # conv1d: conv on last dim
            out = self.conv1d(out)
            out = self.conv1d_post(out)
        if self.is_video():
            out = out.view(out.size(0), -1)
        else:
            out = out.view(features.size(0), -1)

        out = self.classifier(out)
        return out

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):  # pylint: disable=arguments-differ
        dims = len(inputs.shape)

        if self.is_video() and dims == 5:
            batch, times, channels, height, width = inputs.shape
            inputs = inputs.view(batch*times, channels, height, width)

        inputs = self.stem(inputs)

        logits = self.forward_origin(inputs)

        logits /= tau

        if targets is None:
            return logits

        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)

        if self._class_normalize and isinstance(self.loss_fn, (torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth)):
            pos = (targets == 1).to(logits.dtype)
            neg = (targets < 1).to(logits.dtype)
            npos = pos.sum()
            nneg = neg.sum()

            positive_ratio = max(0.1, min(0.9, (npos) / (npos + nneg)))
            negative_ratio = max(0.1, min(0.9, (nneg) / (npos + nneg)))
            LOGGER.debug('[BCEWithLogitsLoss] positive_ratio:%f, negative_ratio:%f',
                         positive_ratio, negative_ratio)

            normalized_loss =  (loss * pos) / positive_ratio
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
        # super(BasicNet, self).half()
        for module in self.modules():
            if len([c for c in module.children()]) > 0:
                continue

            if not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.half()
            else:
                module.float()
        self._half = True
        return self
