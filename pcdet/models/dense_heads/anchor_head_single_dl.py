import numpy as np
import torch.nn as nn
import torch

from .anchor_head_template import AnchorHeadTemplate
from ...utils.loss_utils import QualityFocalLoss


class AnchorHeadSingle_dl(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

        self.print_loss_when_eval = False
        self.f_dense_label = self.model_cfg.get('F_DENSE_LABEL', False) 
        if self.f_dense_label:
            self.quality_loss = QualityFocalLoss()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def get_loss(self, scalar=True):
        cls_loss, tb_dict = self.get_cls_layer_loss(scalar=scalar)
        box_loss, tb_dict_box = self.get_box_reg_layer_loss(scalar=scalar)
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        if scalar:
            tb_dict['rpn_loss'] = rpn_loss.item()
            return rpn_loss, tb_dict
        else:
            tb_dict['rpn_loss'] = rpn_loss
            return cls_loss, box_loss, tb_dict
    
    def forward(self, data_dict, disable_gt_roi_when_pseudo_labeling=False):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C],[2,200,176,18]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        if self.f_dense_label:
            #quality,topk%
            batch_size = cls_preds.shape[0]
            #quality_preds = cls_preds.view(batch_size, -1, self.num_class)[1]
            quality_preds = torch.sigmoid(cls_preds).max(dim=-1)[0][1] #1:unlabelmask
            if self.training and  disable_gt_roi_when_pseudo_labeling:
                print(disable_gt_roi_when_pseudo_labeling)
                quality_preds = quality_preds.flatten(1,-1)
                val, ind = quality_preds.sort(descending=True)
                topk = ind[:,:350].contiguous().view(-1)
                src = val[:,:350].contiguous().view(-1)
                dense_label = torch.zeros_like(quality_preds).view(-1)
                dense_label.scatter_(0,topk,src)
                data_dict['dense_label'] = dense_label
            elif self.training and not disable_gt_roi_when_pseudo_labeling:
                self.forward_ret_dict['dense_label'] = data_dict['dense_label']
                self.forward_ret_dict['dense_preds'] = quality_preds


        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if (self.training or self.print_loss_when_eval) and not disable_gt_roi_when_pseudo_labeling:
            
            
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

    def get_loss(self, scalar=True):
        cls_loss, tb_dict = self.get_cls_layer_loss(scalar=scalar)
        box_loss, tb_dict_box = self.get_box_reg_layer_loss(scalar=scalar)
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        if scalar:
            tb_dict['rpn_loss'] = rpn_loss.item()
            return rpn_loss, tb_dict
        elif self.f_dense_label:
            quality_loss, tb_dict_qu = self.get_quality_loss()
            tb_dict.update(tb_dict_qu)
            return cls_loss, box_loss, quality_loss, tb_dict
        else:
            tb_dict['rpn_loss'] = rpn_loss
            return cls_loss, box_loss, tb_dict
    
    def get_quality_loss(self,):
        dense_label = self.forward_ret_dict['dense_label']
        quality = self.forward_ret_dict['dense_preds'].view(-1)
        positives = dense_label > 0
        negatives = dense_label == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        pos_normalizer = positives.sum(0, keepdim=True).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        qloss = self.quality_loss(quality,dense_label,cls_weights)
        tb_dict = {
            'quality' : qloss.item()
        }
        print(qloss)

        return qloss, tb_dict
    
