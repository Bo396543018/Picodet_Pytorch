# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numbers import Integral

import torch
import torch.nn as nn


from ..builder import BACKBONES


def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self.act = act

        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self._batch_norm = nn.BatchNorm2d(out_channels)

        if self.act == "relu":
            self.activate = nn.ReLU()
        elif self.act == "hard_swish":
            self.activate = nn.Hardswish()


    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act is not None:
            y = self.activate(y)
        return y


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.hardsigmoid(outputs)
        return torch.multiply(inputs, outputs)


class InvertedResidual(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride,
                 act="relu"):
        super(InvertedResidual, self).__init__()
        self._conv_pw = ConvBNLayer(
            in_channels=in_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels // 2,
            act=None)
        self._se = SEModule(mid_channels)

        self._conv_linear = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)

    def forward(self, inputs):
        x1, x2 = torch.split(inputs, split_size_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2], dim=1)
        x2 = self._conv_pw(x2)
        x3 = self._conv_dw(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x3 = self._se(x3)
        x3 = self._conv_linear(x3)
        out = torch.cat([x1, x3], dim=1)
        return channel_shuffle(out, 2)


class InvertedResidualDS(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride,
                 act="relu"):
        super(InvertedResidualDS, self).__init__()

        # branch1
        self._conv_dw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            act=None)
        self._conv_linear_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        # branch2
        self._conv_pw_2 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw_2 = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels // 2,
            act=None)
        self._se = SEModule(mid_channels // 2)
        self._conv_linear_2 = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw_mv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels,
            act="hard_swish")
        self._conv_pw_mv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act="hard_swish")

    def forward(self, inputs):
        x1 = self._conv_dw_1(inputs)
        x1 = self._conv_linear_1(x1)
        x2 = self._conv_pw_2(inputs)
        x2 = self._conv_dw_2(x2)
        x2 = self._se(x2)
        x2 = self._conv_linear_2(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self._conv_dw_mv1(out)
        out = self._conv_pw_mv1(out)

        return out

@BACKBONES.register_module()
class ESNet(nn.Module):
    def __init__(self,
                 model_size="m",
                 activation="hard_swish",
                 out_stages=[4, 11, 14],
                 pretrain=None):
        super(ESNet, self).__init__()

        print("model size is ", model_size)

        if model_size == "s":
            self.scale = 0.75
            self.channel_ratio = [0.875, 0.5, 0.5, 0.5, 0.625, 0.5, 0.625, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        elif model_size == "m":
            self.scale = 1.0
            self.channel_ratio = [0.875, 0.5, 1.0, 0.625, 0.5, 0.75, 0.625, 0.625, 0.5, 0.625, 1.0, 0.625, 0.75]
        elif model_size == "l":
            self.scale = 1.25
            self.channel_ratio = [0.875, 0.5, 1.0, 0.625, 0.5, 0.75, 0.625, 0.625, 0.5, 0.625, 1.0, 0.625, 0.75]
        else:
            raise NotImplementedError

        if isinstance(out_stages, Integral):
            out_stages = [out_stages]
        self.feature_maps = out_stages
        stage_repeats = [3, 7, 3]

        stage_out_channels = [
            -1, 24, make_divisible(128 * self.scale), make_divisible(256 * self.scale),
            make_divisible(512 * self.scale), 1024
        ]

        self._out_channels = []
        self._feature_idx = 0
        # 1. conv1
        self._conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            act=activation)
        self._max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._feature_idx += 1

        # 2. bottleneck sequences
        self._block_list = []
        arch_idx = 0
        for stage_id, num_repeat in enumerate(stage_repeats):
            for i in range(num_repeat):
                channels_scales = self.channel_ratio[arch_idx]
                mid_c = make_divisible(
                    int(stage_out_channels[stage_id + 2] * channels_scales),
                    divisor=8)
                if i == 0:
                    block = InvertedResidualDS(
                        in_channels=stage_out_channels[stage_id + 1],
                        mid_channels=mid_c,
                        out_channels=stage_out_channels[stage_id + 2],
                        stride=2,
                        act=activation)
                else:
                    block = InvertedResidual(
                        in_channels=stage_out_channels[stage_id + 2],
                        mid_channels=mid_c,
                        out_channels=stage_out_channels[stage_id + 2],
                        stride=1,
                        act=activation)
                name = str(stage_id + 2) + '_' + str(i + 1)
                setattr(self, name, block)
                self._block_list.append(block)
                arch_idx += 1
                self._feature_idx += 1
                self._update_out_channels(stage_out_channels[stage_id + 2], self._feature_idx, self.feature_maps)

                self._initialize_weights(pretrain)

    def _update_out_channels(self, channel, feature_idx, feature_maps):
        if feature_idx in feature_maps:
            self._out_channels.append(channel)

    def forward(self, x):
        y = self._conv1(x)
        y = self._max_pool(y)
        outs = []
        for i, inv in enumerate(self._block_list):
            y = inv(y)
            if i + 2 in self.feature_maps:
                outs.append(y)
        return outs

    def _initialize_weights(self, pretrain=True):
        print("init weights...")
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
        if pretrain:
            pretrained_state_dict = torch.load(pretrain)
            print("=> loading pretrained model {}".format(pretrain))
            self.load_state_dict(pretrained_state_dict, strict=False)


if __name__ == "__main__":
    x = torch.zeros((1, 3, 224, 224))
    model = ESNet(model_size='m')
    print(model)

    out = model(x)

    print([sub.shape for sub in out]) # out channels

    # out channels
    # s: [96, 192, 384]
    # m: [128, 256, 512]
    # l: [160, 320, 640]