import torch

from ...utils import box_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from .point_head_template import PointHeadTemplate
from ...utils.loss_utils import QualityFocalLoss


class PointHeadSimple_dl(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PV-RCNN keypoint segmentaion.
    Reference Paper: https://arxiv.org/abs/1912.13192
    PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
    """
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

        self.print_loss_when_eval = False
        self.dense_label = self.model_cfg.get('DENSE_LABEL', False)
        if self.dense_label:
            self.dense_loss_func = QualityFocalLoss()

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        if not self.dense_label:
            targets_dict = self.assign_stack_targets(
                points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=True, use_ball_constraint=False,
                ret_part_labels=False
                )
        else:
            gt_boxes_dense = input_dict['gt_dense_label']
            #label_dense = gt_boxes.new_zeros(gt_boxes.shape[1]).unsqueeze(dim=0)
            #unlabel_dense = torch.cat([unlabel_dense,unlabel_dense.new_zeros((gt_boxes.shape[1] - unlabel_dense.shape[0]))], dim=-1).unsqueeze(dim=0)
            #gt_boxes_dense = torch.cat([label_dense, unlabel_dense],dim=0)
            targets_dict = self.assign_stack_targets_dense(
                points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=True, use_ball_constraint=False,
                ret_part_labels=False, gt_boxes_dense=gt_boxes_dense
                )

        return targets_dict

    def get_loss(self, tb_dict=None, scalar=True):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss(None, scalar=scalar)

        tb_dict.update(tb_dict_1)
        return point_loss_cls, tb_dict

    def forward(self, batch_dict, disable_gt_roi_when_pseudo_labeling=False):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)

        ret_dict = {
            'point_cls_preds': point_cls_preds,
        }

        point_cls_scores = torch.sigmoid(point_cls_preds)
        batch_dict['point_cls_scores'], _ = point_cls_scores.max(dim=-1)

        # should not use gt_roi for pseudo label generation
        if (self.training or self.print_loss_when_eval) and not disable_gt_roi_when_pseudo_labeling:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            if self.dense_label:
                ret_dict['point_dense_labels'] = targets_dict['point_dense_labels']
        self.forward_ret_dict = ret_dict

        return batch_dict

    def assign_stack_targets_dense(self, points, gt_boxes, gt_boxes_dense,extend_gt_boxes=None,
                             ret_box_labels=False, ret_part_labels=False,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        point_dense_labels = points.new_zeros(points.shape[0])
        
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            point_dense_labels_single = point_dense_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
                #point_dense_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

            
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single
            gt_box_of_fg_points = gt_boxes_dense[k][box_idxs_of_pts[fg_flag]]
            point_dense_labels_single[fg_flag] = gt_box_of_fg_points
            point_dense_labels[bs_mask] = point_dense_labels_single


        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'point_part_labels': point_part_labels,
            'point_dense_labels':point_dense_labels,
        }
        return targets_dict

    def dense_loss(self,tb_dict):
        
        dense_labels = self.forward_ret_dict['point_dense_labels'].view(-1)
        dense_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class)
        positives = dense_labels > 0
        negative_dense_weights = (dense_labels == 0) * 1.0
        dense_weights = (negative_dense_weights + 1.0 * positives).float()
        
        batch_size = 2
        pos_normalizer = positives.reshape(batch_size, -1).sum(dim=1, keepdim=True).float()
        dense_weights = dense_weights.reshape(batch_size, -1)
        dense_weights /= torch.clamp(pos_normalizer, min=1.0)
        dense_weights = dense_weights.reshape(-1)

        dense_targets = dense_preds.new_zeros(*list(dense_labels.shape), self.num_class+1)
        dense_targets.scatter_(-1, torch.ceil(dense_labels).long().unsqueeze(dim=-1), dense_labels.view(-1,1))
        dense_targets = dense_targets[..., 1:]
        dense_loss = self.dense_loss_func(dense_preds, dense_targets,dense_weights)
        dense_loss = dense_loss.reshape(batch_size,-1).sum(-1)
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({"dense_loss":dense_loss})


        return dense_loss, tb_dict