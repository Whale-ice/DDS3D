import torch
import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
from ...utils import loss_utils


class AnchorHeadSingle_db(AnchorHeadTemplate):
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
        self.use_dense = self.model_cfg.get('USE_DENSE', False)
        self.quality_loss = loss_utils.QualityFocalLoss()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict, disable_gt_roi_when_pseudo_labeling=False):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if (self.training or self.print_loss_when_eval) and not disable_gt_roi_when_pseudo_labeling:
            if self.use_dense:
                targets_dict = self.assign_targets_dense(
                    gt_boxes=data_dict['gt_boxes'], densescores = data_dict['densescore']
                    )
            else:
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
        dense_loss,tb_dict_dense = self.get_dense_cls_layer_loss(scalar=scalar)
        tb_dict.update(tb_dict_box)
        tb_dict.update(tb_dict_dense)
        rpn_loss = cls_loss + box_loss

        if scalar:
            tb_dict['rpn_loss'] = rpn_loss.item()
            return rpn_loss, tb_dict
        else:
            tb_dict['rpn_loss'] = rpn_loss
            return cls_loss, box_loss, dense_loss, tb_dict
    
    def get_dense_cls_layer_loss(self, scalar=True):
        #only consider unlabeled like soft teacher
        import pdb
        pdb.set_trace()
        cls_preds = self.forward_ret_dict['cls_preds'][1:,...]
        box_cls_labels = self.forward_ret_dict['box_cls_labels'][1:,...] #(b, 2*3*h*w)
        dense_scores = self.forward_ret_dict['dense_score']
        batch_size = 1
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()

        #print(positives,negative_cls_weights,cls_weights,cls_weights.max())
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1


        pos_normalizer = positives.sum(1, keepdim=True).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), dense_scores.unsqueeze(dim=-1))
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.quality_loss(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        if scalar:
            cls_loss = cls_loss_src.sum() / batch_size
            rpn_acc_cls = ((cls_preds.max(-1)[1] + 1) == cls_targets.long()).sum().float() / \
                          torch.clamp((cls_targets > 0).sum().float(), min=1.0)
        else:
            cls_loss = cls_loss_src.reshape(batch_size, -1).sum(-1)
            rpn_acc_cls = ((cls_preds.max(-1)[1] + 1) == cls_targets.long()).view(batch_size, -1).sum(-1).float() / \
                          torch.clamp((cls_targets > 0).view(batch_size, -1).sum(-1).float(), min=1.0)

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

        tb_dict = {
            'dense': cls_loss.item() if scalar else cls_loss,
            'dense_a_c_c': rpn_acc_cls.item() if scalar else rpn_acc_cls
        }

        return cls_loss, tb_dict

    def assign_targets_dense(self, gt_boxes, densescores):
        """
        Args:
            gt_boxes: (B, M, 8+1)
        Returns:

        """
        densescores = densescores.view(1,-1,1)
        targets_dict = self.target_assigner.assign_targets_dense(
            self.anchors, gt_boxes,densescores
        )
        return targets_dict
