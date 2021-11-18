# Copyright 2021.
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


import torch
import torch.nn as nn

from mmcv.runner import force_fp32
from mmdet.core import (MlvlPointGenerator, multi_apply, build_assigner, build_sampler, 
                    distance2bbox, bbox2distance, images_to_levels, reduce_mean)
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl

from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from .gfl_head import Integral
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead


@HEADS.register_module()
class PicoDetHead(AnchorFreeHead):
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
        super(PicoDetHead, self).__init__(
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
        for _ in self.strides:
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


    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        num_imgs = len(img_metas)
        
        num_level_anchors = [ 
                featmap.shape[-2] * featmap.shape[-1] for featmap in cls_scores]
                 
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, cls_scores[0].device, with_stride=True)

        # 所有batch结果拼在了一起
        decode_bbox_preds = []
        center_and_strides = []
        for stride, bbox_pred, center_and_stride in zip(self.strides, bbox_preds, mlvl_priors):
            # h w 2
            center_and_stride = center_and_stride.repeat(num_imgs, 1, 1)
            center_and_strides.append(center_and_stride)
            center_in_feature = center_and_stride.reshape(
                (-1, 4))[:, :-2] / stride
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * (self.reg_max + 1))
            pred_corners = self.integral(bbox_pred)
            decode_bbox_pred = distance2bbox(
                center_in_feature, pred_corners).reshape(num_imgs, -1, 4)
            decode_bbox_preds.append(decode_bbox_pred * stride) # 将框的结果转换成xyxy,采用的是逐level转换

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(
                [num_imgs, -1, self.cls_out_channels])
            for cls_pred in cls_scores
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1) # N x #points x #cls_out_channels
        flatten_bboxes = torch.cat(decode_bbox_preds, dim=1)
        flatten_center_and_strides = torch.cat(center_and_strides, dim=1)

        pos_num_l, label_l, label_weight_l, bbox_target_l = [], [], [], []

        for flatten_cls_pred, flatten_center_and_stride, flatten_bbox, gt_bbox, gt_label \
                    in zip(flatten_cls_preds.detach(), flatten_center_and_strides.detach(), \
                        flatten_bboxes.detach(), gt_bboxes, gt_labels): # 逐图片
            pos_num, label, label_weight, bbox_target = self._get_target_single(
                flatten_cls_pred, flatten_center_and_stride, flatten_bbox, 
                gt_bbox, gt_label)
            pos_num_l.append(pos_num)
            label_l.append(label)
            label_weight_l.append(label_weight)
            bbox_target_l.append(bbox_target)



        center_and_strides_list = images_to_levels([flatten_cs for flatten_cs in flatten_center_and_strides], 
                                                        num_level_anchors)
        labels_list = images_to_levels(label_l, num_level_anchors)
        label_weights_list = images_to_levels(label_weight_l,
                                                    num_level_anchors)
        bbox_targets_list = images_to_levels(bbox_target_l,
                                                   num_level_anchors)

        num_total_pos = sum([max(pos_num, 1) for pos_num in pos_num_l])        # sync num_pos across all gpus

        if self.sync_num_pos:
            num_pos_avg_per_gpu = reduce_mean(
                labels_list[0].new_tensor(num_total_pos).float()).item()
            num_pos_avg_per_gpu = max(num_pos_avg_per_gpu, 1.0)
        else:
            num_pos_avg_per_gpu = num_total_pos
        # 这里应该能用get_loss_single实现, 逐level
        loss_bbox_list, loss_dfl_list, loss_cls_list, avg_factor = [], [], [], []
        for cls_score, bbox_pred, center_and_strides, labels, label_weights, bbox_targets, stride in zip(
                cls_scores, bbox_preds, center_and_strides_list, labels_list,
                label_weights_list, bbox_targets_list, self.strides):
            center_and_strides = center_and_strides.reshape(-1, 4)
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(
                [-1, self.cls_out_channels]
            )
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(
                -1, 4 * (self.reg_max + 1)
            )
            bbox_targets = bbox_targets.reshape(-1, 4)
            labels = labels.reshape(-1)
            label_weights = label_weights.reshape(-1)

            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            bg_class_ind = self.num_classes
            # pos_inds = ((labels >= 0)
            #             & (labels < bg_class_ind)).nonzero().squeeze(1)
            pos_inds = torch.nonzero((labels >= 0)  & (labels < bg_class_ind)).squeeze(1)

            if self.use_vfl:
                score = label_weights.new_zeros(cls_score.shape)
            else:
                score = label_weights.new_zeros(labels.shape)

            if num_total_pos > 0:
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_bbox_pred = bbox_pred[pos_inds]
                pos_centers = center_and_strides[:, :-2][pos_inds] / stride

                weight_targets = cls_score.detach().sigmoid()
                weight_targets = weight_targets.max(dim=1)[0][pos_inds]
                pos_bbox_pred_corners = self.integral(pos_bbox_pred)
                pos_decode_bbox_pred = distance2bbox(pos_centers,
                                                    pos_bbox_pred_corners)
                pos_decode_bbox_targets = pos_bbox_targets / stride
                
                pos_ious = bbox_overlaps(
                        pos_decode_bbox_pred,
                        pos_decode_bbox_targets.detach(),
                        is_aligned=True).clamp(min=1e-6)
                if self.use_vfl:
                    pos_labels = labels[pos_inds]
                    score[pos_inds, pos_labels] = pos_ious.clone().detach()
                else:
                    score[pos_inds] = pos_ious.clone().detach()
                pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
                target_corners = bbox2distance(pos_centers,
                                            pos_decode_bbox_targets,
                                            self.reg_max).reshape(-1)
                # regression loss
                loss_bbox = self.loss_bbox(
                    pos_decode_bbox_pred,
                    pos_decode_bbox_targets.detach(),
                    weight=weight_targets,
                    avg_factor=1.0)
                # dfl loss
                loss_dfl = self.loss_dfl(
                    pred_corners,
                    target_corners,
                    weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                    avg_factor=4.0
                )
            else:
                loss_bbox = bbox_pred.sum() * 0
                loss_dfl = bbox_pred.sum() * 0
                weight_targets = bbox_pred.new_tensor(0)
            # cls (qfl) loss
            if self.use_vfl:
                loss_cls = self.loss_cls(
                    cls_score, 
                    score, 
                    avg_factor=num_pos_avg_per_gpu)
            else:
                loss_cls = self.loss_cls(
                    cls_score, (labels, score),
                    weight=label_weights,
                    avg_factor=num_pos_avg_per_gpu
                )

            loss_bbox_list.append(loss_bbox)
            loss_dfl_list.append(loss_dfl)
            loss_cls_list.append(loss_cls)
            avg_factor.append(weight_targets.sum())

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / avg_factor, loss_bbox_list))
        losses_dfl = list(map(lambda x: x / avg_factor, loss_dfl_list))
        losses_cls = sum(loss_cls_list)
  

        losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

        return losses
        
    @torch.no_grad()
    def _get_target_single(self, cls_preds, priors, decoded_bboxes,
                           gt_bboxes, gt_labels):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        num_bboxes = decoded_bboxes.shape[0]
        # # No target
        if num_gts == 0:
            labels = priors.new_full((num_bboxes, ),
                                  self.num_classes,
                                  dtype=torch.long)
            label_weights = priors.new_zeros(num_bboxes, dtype=torch.float)


            bbox_targets = torch.zeros_like(decoded_bboxes) # anchors 在nanodet称为grid_cells
            # foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (0, labels, label_weights, bbox_targets)

         # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        # offset_priors = torch.cat(
        #     [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)
        assign_result = self.assigner.assign(
            cls_preds.sigmoid(),
            priors, decoded_bboxes, gt_bboxes, gt_labels)

        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        num_pos_per_img = pos_inds.size(0)
        bbox_targets = torch.zeros_like(decoded_bboxes) # anchors 在nanodet称为grid_cells
        bbox_weights = torch.zeros_like(decoded_bboxes)
        labels = priors.new_full((num_bboxes, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = priors.new_zeros(num_bboxes, dtype=torch.float)

        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (num_pos_per_img, labels, label_weights, bbox_targets)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """Transform network outputs of a batch into bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Default None.
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].device,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs)
            result_list.append(results)
        return result_list
   
    def get_targets(self, ):

        pass

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. GFL head does not need this value.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, stride, priors) in enumerate(
                zip(cls_score_list, bbox_pred_list,
                    self.prior_generator.strides, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert stride[0] == stride[1]

            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.integral(bbox_pred) * stride[0]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            bboxes = distance2bbox(
                        priors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        return self._bbox_post_process(
            mlvl_scores,
            mlvl_labels,
            mlvl_bboxes,
            img_meta['scale_factor'],
            cfg,
            rescale=rescale,
            with_nms=with_nms)

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
  
