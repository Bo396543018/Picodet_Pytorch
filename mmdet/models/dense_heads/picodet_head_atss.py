# Copyright 2021 Bo Chen.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmcv.runner import force_fp32
from mmdet.core import (MlvlPointGenerator, multi_apply, build_assigner, build_sampler, 
                    distance2bbox, bbox2distance, images_to_levels, reduce_mean)
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl

from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from .gfl_head import Integral
from ..builder import HEADS, build_loss


from .gfl_head import GFLHead, Integral

@HEADS.register_module()
class PicoDetATSSHead(GFLHead):
    """
    Modified from GFL, use same loss functions but much lightweight convolution heads
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels=96,
        stacked_convs=2,
        kernel_size=5,
        share_cls_reg=False,
        conv_type="DWConv",
        conv_cfg=None,
        norm_cfg=None,
        use_vfl=False,
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=7,
        strides=[8, 16, 32],
        activation="LeakyReLU",
        sync_num_pos=True,
        init_cfg=dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal',
                name='gfl_cls',
                std=0.01,
                bias_prob=0.01)),
        **kwargs):
        self.kernel_size = kernel_size
        self.conv_cfg = conv_cfg
        self.reg_max = reg_max
        self.share_cls_reg = share_cls_reg
        self.activation = activation
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule
        super(PicoDetATSSHead, self).__init__(
            num_classes, in_channels, stacked_convs=stacked_convs, strides=strides,
            feat_channels=feat_channels, loss_cls=loss_cls, norm_cfg=norm_cfg, init_cfg=init_cfg, **kwargs)
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.prior_generator = MlvlPointGenerator(strides, offset=0.5)
        self.sync_num_pos = sync_num_pos
        self.integral = Integral(self.reg_max)
        self.loss_dfl = build_loss(loss_dfl)
        self.use_vfl = use_vfl
        if use_vfl:
            assert loss_cls['type'] == 'VarifocalLoss', 'when set use_vfl=True, loss_cls type must be VarifocalLoss'
            self.loss_cls = build_loss(loss_cls)

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in self.strides: # 每个head都有一个
            cls_convs, reg_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels,
                    self.cls_out_channels + 4 * (self.reg_max + 1)
                    if self.share_cls_reg
                    else self.cls_out_channels,
                    1,
                    padding=0,
                )
                for _ in self.strides
            ]
        )
        # TODO: if
        # self.gfl_reg = None
        # if not self.share_cls_reg:
        self.gfl_reg = nn.ModuleList(
            [
                nn.Conv2d(self.feat_channels, 4 * (self.reg_max + 1), 1, padding=0)
                for _ in self.strides
            ]
        )

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    self.kernel_size,
                    stride=1,
                    padding=(self.kernel_size - 1) // 2,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
            if not self.share_cls_reg:
                reg_convs.append(
                    self.ConvModule(
                        chn,
                        self.feat_channels,
                        self.kernel_size,
                        stride=1,
                        padding=(self.kernel_size - 1) // 2,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None,
                        activation=self.activation,
                    )
                )

        return cls_convs, reg_convs

    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)
        print("Finish initialize PicoDet Head.")

    def forward(self, feats):
        return multi_apply(
            self.forward_single,
            feats,
            self.cls_convs,
            self.reg_convs,
            self.gfl_cls,
            self.gfl_reg,
        )

    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg):
        cls_feat = x
        reg_feat = x
        for cls_conv in cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in reg_convs:
            reg_feat = reg_conv(reg_feat)
        if self.share_cls_reg:
            feat = gfl_cls(cls_feat)
            cls_score, bbox_pred = torch.split(
                feat, [self.cls_out_channels, 4 * (self.reg_max + 1)], dim=1
            )
        else:
            cls_score = gfl_cls(cls_feat)
            bbox_pred = gfl_reg(reg_feat)

        if torch.onnx.is_in_onnx_export():
            cls_score = (
                torch.sigmoid(cls_score)
                .reshape(1, self.num_classes, -1)
                .permute(0, 2, 1)
            )
            bbox_pred = bbox_pred.reshape(1, (self.reg_max + 1) * 4, -1).permute(
                0, 2, 1
            )
        return cls_score, bbox_pred

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Hack some keys of the model state dict so that can load checkpoints
        of previous version."""
        version = local_metadata.get('version', None)
        version = 'ppdet2mmdet'
        local_metadata['version'] = 'ppdet2mmdet'
        if version is None:
            # the key is different in early versions
            # for example, 'fcos_cls' become 'conv_cls' now
            bbox_head_keys = [
                k for k in state_dict.keys() if k.startswith(prefix)
            ]
            ori_predictor_keys = []
            new_predictor_keys = []
            # e.g. 'fcos_cls' or 'fcos_reg'
            for key in bbox_head_keys:
                ori_predictor_keys.append(key)
                key = key.split('.')
                conv_name = None
                if key[1].endswith('cls'):
                    conv_name = 'conv_cls'
                elif key[1].endswith('reg'):
                    conv_name = 'conv_reg'
                elif key[1].endswith('centerness'):
                    conv_name = 'conv_centerness'
                else:
                    assert NotImplementedError
                if conv_name is not None:
                    key[1] = conv_name
                    new_predictor_keys.append('.'.join(key))
                else:
                    ori_predictor_keys.pop(-1)
            for i in range(len(new_predictor_keys)):
                state_dict[new_predictor_keys[i]] = state_dict.pop(
                    ori_predictor_keys[i])
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)
  
