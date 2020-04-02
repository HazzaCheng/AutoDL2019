import logging
import sys

import torch
from torch.utils import model_zoo
from .efficientnet_pytorch.model import EfficientNet
from .efficientnet_pytorch.utils import get_model_params, round_filters, url_map

from ..import skeleton

formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s %(filename)s] %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(handler)


class EfficientNetB0(EfficientNet):

    def __init__(self, in_channels, num_classes=10, **kwargs):
        blocks_args, global_params = get_model_params('efficientnet-b0', None)
        super(EfficientNetB0, self).__init__(blocks_args, global_params)

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

        out_channels = round_filters(1280, self._global_params)
        self._fc = torch.nn.Linear(out_channels, num_classes)

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
        sd = model_zoo.load_url(url_map['efficientnet-b0'], model_dir=model_dir)
        del sd['_fc.weight']
        del sd['_fc.bias']

        self.load_state_dict(sd, strict=False)

        for idx in range(len(self.stem)):
            m = self.stem[idx]
            if hasattr(m, 'weight'):
                torch.nn.init.xavier_normal_(m.weight, gain=gain)
                LOGGER.debug('initialize stem weight')

        torch.nn.init.xavier_uniform_(self._fc.weight, gain=gain)
        LOGGER.debug('initialize classifier weight')

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):
        dims = len(inputs.shape)

        if self.is_video() and dims == 5:
            batch, times, channels, height, width = inputs.shape
            inputs = inputs.view(batch * times, channels, height, width)

        inputs = self.stem(inputs)
        logits = EfficientNet.forward(self, inputs)
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
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
                continue
            convert_module(module, dtype=torch.half)
            if isinstance(module, torch.nn.RNNBase) or isinstance(module, torch.nn.modules.rnn.RNNBase):
                module.flatten_parameters()

        self._half = True

        return self


def convert_module(module, dtype):
    """
    Converts a module's immediate parameters and buffers to dtype.
    """
    for param in module.parameters(recurse=False):
        if param is not None:
            if param.data.dtype.is_floating_point:
                param.data = param.data.to(dtype=dtype)
            if param._grad is not None and param._grad.data.dtype.is_floating_point:
                param._grad.data = param._grad.data.to(dtype=dtype)

    for buf in module.buffers(recurse=False):
        if buf is not None and buf.data.dtype.is_floating_point:
            buf.data = buf.data.to(dtype=dtype)


class EfficientNetB1(EfficientNet):

    def __init__(self, in_channels, num_classes=10, **kwargs):
        blocks_args, global_params = get_model_params('efficientnet-b1', None)
        super(EfficientNetB1, self).__init__(blocks_args, global_params)

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

        out_channels = round_filters(1280, self._global_params)
        self._fc = torch.nn.Linear(out_channels, num_classes)

        self._half = False
        self._class_normalize = True

    def init(self, model_dir, gain=1.):
        sd = model_zoo.load_url(url_map['efficientnet-b1'], model_dir=model_dir)
        del sd['_fc.weight']
        del sd['_fc.bias']

        self.load_state_dict(sd, strict=False)

        for idx in range(len(self.stem)):
            m = self.stem[idx]
            if hasattr(m, 'weight'):
                torch.nn.init.xavier_normal_(m.weight, gain=gain)
                LOGGER.debug('initialize stem weight')

        torch.nn.init.xavier_uniform_(self._fc.weight, gain=gain)
        LOGGER.debug('initialize classifier weight')

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):  # pylint: disable=arguments-differ
        inputs = self.stem(inputs)
        logits = EfficientNet.forward(self, inputs)
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
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
                continue
            convert_module(module, dtype=torch.half)
            if isinstance(module, torch.nn.RNNBase) or isinstance(module, torch.nn.modules.rnn.RNNBase):
                module.flatten_parameters()

        self._half = True

        return self

class EfficientNetB2(EfficientNet):

    def __init__(self, in_channels, num_classes=10, **kwargs):
        blocks_args, global_params = get_model_params('efficientnet-b2', None)
        super(EfficientNetB2, self).__init__(blocks_args, global_params)

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

        out_channels = round_filters(1280, self._global_params)
        self._fc = torch.nn.Linear(out_channels, num_classes)

        self._half = False
        self._class_normalize = True

    def init(self, model_dir, gain=1.):
        sd = model_zoo.load_url(url_map['efficientnet-b2'], model_dir=model_dir)
        del sd['_fc.weight']
        del sd['_fc.bias']

        self.load_state_dict(sd, strict=False)

        for idx in range(len(self.stem)):
            m = self.stem[idx]
            if hasattr(m, 'weight'):
                torch.nn.init.xavier_normal_(m.weight, gain=gain)
                LOGGER.debug('initialize stem weight')

        torch.nn.init.xavier_uniform_(self._fc.weight, gain=gain)
        LOGGER.debug('initialize classifier weight')

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):
        inputs = self.stem(inputs)
        logits = EfficientNet.forward(self, inputs)
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
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
                continue
            convert_module(module, dtype=torch.half)
            if isinstance(module, torch.nn.RNNBase) or isinstance(module, torch.nn.modules.rnn.RNNBase):
                module.flatten_parameters()

        self._half = True

        return self
