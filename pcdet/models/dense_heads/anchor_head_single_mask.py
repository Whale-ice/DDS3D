import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle_mask(AnchorHeadTemplate):
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
            print(data_dict['use_weak'])
            if data_dict['use_weak'] :
                targets_dict = self.assign_targets_mask(
                    gt_boxes=data_dict['gt_boxes'],
                    weak_boxes= data_dict['weak_boxes']
                )
                self.forward_ret_dict.update(targets_dict)
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
    
    def assign_targets_mask(self, gt_boxes, weak_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        #import torch
        #torch.cuda.synchronize()
        #import time
        #start = time.time()
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        ign = targets_dict['box_cls_labels'][1]<0
        neg = targets_dict['box_cls_labels'][1]==0
        pos = targets_dict['box_cls_labels'][1]>0
        weak_targets_dict = self.target_assigner.assign_targets(
            self.anchors, weak_boxes
        )
        ign1 =weak_targets_dict['box_cls_labels'][1]<0
        neg1 =weak_targets_dict['box_cls_labels'][1]==0
        pos1 =weak_targets_dict['box_cls_labels'][1]>0
        import pdb
        pdb.set_trace()
        targets_dict['box_cls_labels'][1][pos1] = -1
        #end = time.time()
        #print(end-start)
        
        return targets_dict
