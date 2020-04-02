#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-03-05
import logging
import math
import sys
from collections import OrderedDict

import torch
from torch import nn
from torch.utils import model_zoo

from .. import skeleton

model_urls = {
    'se_resnext50_32x4d': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
    'se_resnext101_32x4d': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
}

formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s %(filename)s] %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(handler)


class SeResNext50(nn.Module):

    def __init__(self, in_channels, num_classes, **kwargs):
    # def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
    #              inplanes=128, input_3x3=True, downsample_kernel_size=3,
    #              downsample_padding=1, num_classes=1000):

    # SEResNeXtBottleneck, [3, 4, 23, 3], groups = 32, reduction = 16,
    # dropout_p = None, inplanes = 64, input_3x3 = False,
    # downsample_kernel_size = 1, downsample_padding = 0,
    # num_classes = num_classes
        super(SeResNext50, self).__init__()

        block = SEResNeXtBottleneck
        layers = [3, 4, 6, 3]
        groups = 32
        reduction = 16
        dropout_p = None
        inplanes = 64
        input_3x3 = False
        downsample_kernel_size = 1
        downsample_padding = 0

        self.inplanes = inplanes

        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(self.inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )

        self.last_channels = 512 * block.expansion
        # self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(self.last_channels, num_classes)

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

        self._class_normalize = True
        self._is_video = False
        self._half = False

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
        sd = model_zoo.load_url(model_urls['se_resnext50_32x4d'], model_dir=self.model_dir)
        # sd = model_zoo.load_url(model_urls['resnet34'], model_dir='./models/')
        del sd['last_linear.weight']
        del sd['last_linear.bias']
        self.load_state_dict(sd, strict=False)

        torch.nn.init.xavier_uniform_(self.last_linear.weight, gain=gain)
        LOGGER.debug('initialize classifier weight')

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)

        if self.is_video():
            x = self.conv1d_prev(x)
            x = x.view(x.size(0), x.size(1), -1)
            x = self.conv1d(x)
            x = self.conv1d_post(x)

        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward_origin(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):
        dims = len(inputs.shape)

        if self.is_video() and dims == 5:
            batch, times, channels, height, width = inputs.shape
            inputs = inputs.view(batch * times, channels, height, width)

        inputs = self.stem(inputs)
        logits = self.forward_origin(inputs)
        logits /= tau

        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)

        if self._class_normalize and isinstance(self.loss_fn, (
                torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth)):
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


class SeResNext101(nn.Module):

    def __init__(self, in_channels, num_classes, **kwargs):
    # def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
    #              inplanes=128, input_3x3=True, downsample_kernel_size=3,
    #              downsample_padding=1, num_classes=1000):

    # SEResNeXtBottleneck, [3, 4, 23, 3], groups = 32, reduction = 16,
    # dropout_p = None, inplanes = 64, input_3x3 = False,
    # downsample_kernel_size = 1, downsample_padding = 0,
    # num_classes = num_classes
        super(SeResNext101, self).__init__()

        block = SEResNeXtBottleneck
        layers = [3, 4, 23, 3]
        groups = 32
        reduction = 16
        dropout_p = None
        inplanes = 64
        input_3x3 = False
        downsample_kernel_size = 1
        downsample_padding = 0

        self.inplanes = inplanes

        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(self.inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )

        self.last_channels = 512 * block.expansion
        # self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

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

        self._class_normalize = True
        self._is_video = False
        self._half = False

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
        sd = model_zoo.load_url(model_urls['se_resnext101_32x4d'], model_dir=self.model_dir)
        # sd = model_zoo.load_url(model_urls['resnet34'], model_dir='./models/')
        del sd['last_linear.weight']
        del sd['last_linear.bias']
        self.load_state_dict(sd, strict=False)

        torch.nn.init.xavier_uniform_(self.last_linear.weight, gain=gain)
        LOGGER.debug('initialize classifier weight')

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)

        if self.is_video():
            x = self.conv1d_prev(x)
            x = x.view(x.size(0), x.size(1), -1)
            x = self.conv1d(x)
            x = self.conv1d_post(x)

        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward_origin(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):
        dims = len(inputs.shape)

        if self.is_video() and dims == 5:
            batch, times, channels, height, width = inputs.shape
            inputs = inputs.view(batch * times, channels, height, width)

        inputs = self.stem(inputs)
        logits = self.forward_origin(inputs)
        logits /= tau

        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)

        if self._class_normalize and isinstance(self.loss_fn, (
                torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth)):
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


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride
