import torch

from ...utils import box_utils
from .point_head_template import PointHeadTemplate


class PointHeadSimple(PointHeadTemplate):
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
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False
        )

        return targets_dict

    def get_loss(self, tb_dict=None, scalar=True):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss(None, scalar=scalar)

        point_loss = point_loss_cls
        tb_dict.update(tb_dict_1)
        return point_loss, tb_dict

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
        self.forward_ret_dict = ret_dict

        return batch_dict
    
    def get_cls_layer_loss_1(self, tb_dict=None, scalar=True):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class)

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()

        if scalar:
            pos_normalizer = positives.sum(dim=0).float()
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        else:
            batch_size = 2 # TODO this is hardcoded now
            pos_normalizer = positives.reshape(batch_size, -1).sum(dim=1, keepdim=True).float()
            cls_weights = cls_weights.reshape(batch_size, -1)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_weights = cls_weights.reshape(-1)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        #pointrcnn 加一个操作
        point_cls_preds = point_cls_preds.max(-1)[0].float()
        if scalar:
            point_loss_cls = cls_loss_src.sum()
            point_acc_cls = ((torch.sigmoid(point_cls_preds) > 0.5).long() == point_cls_labels.long()).float().mean()
        else:
            point_loss_cls = cls_loss_src.reshape(batch_size, -1).sum(-1)
            point_acc_cls = ((torch.sigmoid(point_cls_preds) > 0.5).long() == point_cls_labels.long()).float().view(batch_size, -1).mean(-1)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'point_loss_cls': point_loss_cls.item() if scalar else point_loss_cls,
            'point_pos_num': pos_normalizer.item() if scalar else pos_normalizer,
            'point_acc_cls': point_acc_cls.item() if scalar else point_acc_cls
        })
        return point_loss_cls, tb_dict