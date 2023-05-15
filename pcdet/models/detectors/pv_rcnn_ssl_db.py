import os

import torch
import copy
import numpy as np

from pcdet.datasets.augmentor.augmentor_utils import *
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from .detector3d_template import Detector3DTemplate
from.pv_rcnn import PVRCNN


class PVRCNN_SSL_db(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # something changes so need deep copy
        model_cfg_copy = copy.deepcopy(model_cfg)
        dataset_copy = copy.deepcopy(dataset)
        self.pv_rcnn = PVRCNN(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

#  teacher
        self.pv_rcnn_ema = PVRCNN(model_cfg=model_cfg_copy, num_class=num_class, dataset=dataset_copy)
        for param in self.pv_rcnn_ema.parameters():
            param.detach_()
        self.add_module('pv_rcnn', self.pv_rcnn)
        self.add_module('pv_rcnn_ema', self.pv_rcnn_ema)

        # self.module_list = self.build_networks()
        # self.module_list_ema = self.build_networks()
        self.thresh = model_cfg.THRESH
        self.sem_thresh = model_cfg.SEM_THRESH
        self.unlabeled_supervise_cls = model_cfg.UNLABELED_SUPERVISE_CLS
        self.unlabeled_supervise_refine = model_cfg.UNLABELED_SUPERVISE_REFINE
        self.unlabeled_weight = model_cfg.UNLABELED_WEIGHT
        self.decay_step = model_cfg.DECAY_STEP
        self.minthr = model_cfg.MINTHR
        self.register_buffer('step', torch.LongTensor(1).zero_())

    def forward(self, batch_dict):
        if self.training:
            batch_dict['mask'] = batch_dict['mask'][:,:1]
            mask = batch_dict['mask'].view(-1)

            labeled_mask = torch.nonzero(mask).squeeze(1).long()
            unlabeled_mask = torch.nonzero(1-mask).squeeze(1).long()
            batch_dict_ema = {}
            keys = list(batch_dict.keys())
            for k in keys:
                if k + '_ema' in keys:
                    continue
                if k.endswith('_ema'):
                    batch_dict_ema[k[:-4]] = batch_dict[k]
                else:
                    batch_dict_ema[k] = batch_dict[k]

            with torch.no_grad():
                # self.pv_rcnn_ema.eval()  # Important! must be in train mode
                for cur_module in self.pv_rcnn_ema.module_list:
                    try:
                        batch_dict_ema = cur_module(batch_dict_ema, disable_gt_roi_when_pseudo_labeling=True)
                    except:
                        batch_dict_ema = cur_module(batch_dict_ema)  

                pred_dicts = []
                pseudo_boxes = []
                pseudo_scores = []
                pseudo_labels = []
                max_box_num = batch_dict['gt_boxes'].shape[1]
                max_pseudo_box_num = 0

                for ind in unlabeled_mask:
                    box_preds = batch_dict_ema['batch_box_preds'][ind]
                    cls_preds = batch_dict_ema['batch_cls_preds'][ind]
                    cls_preds = torch.sigmoid(cls_preds)
                    cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                    label_preds = batch_dict_ema['roi_labels'][ind]
                    sem_scores = batch_dict_ema['roi_scores'][ind]
                    final_sem_scores = torch.sigmoid(sem_scores)
                    pred_dicts = {
                        'pred_boxes': box_preds,
                        'pred_scores': cls_preds,
                        'pred_labels': label_preds,
                        'pred_sem_scores': final_sem_scores
                    }
                    pseudo_score = pred_dicts['pred_scores']
                    pseudo_box = pred_dicts['pred_boxes']
                    pseudo_label = pred_dicts['pred_labels']
                    pseudo_sem_score = pred_dicts['pred_sem_scores']

                    if len(pseudo_label) == 0:
                        pseudo_boxes.append(pseudo_label.new_zeros((0, 8)).float())
                        num=0
                        semnum=0
                        continue

                    conf_thresh = torch.tensor(self.sem_thresh,device=pseudo_label.device)
                    conf_thresh = conf_thresh.unsqueeze(0).repeat(len(pseudo_label),1)
                    conf_thresh = conf_thresh.gather(dim=1,index=(pseudo_label-1).unsqueeze(-1))
            
                    #valid_inds = pseudo_score > conf_thresh.squeeze()
                    #scorenum = valid_inds.sum()

                    valid_inds = pseudo_sem_score > conf_thresh.squeeze()
                    num = valid_inds.sum()
                    semnum = (pseudo_sem_score > self.sem_thresh[0]).sum()
                   
                    pseudo_sem_score = pseudo_sem_score[valid_inds]
                    pseudo_box = pseudo_box[valid_inds]
                    pseudo_label = pseudo_label[valid_inds]
                    pseudo_score = pseudo_score[valid_inds]

                    pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
                    
                    if pseudo_box.shape[0] > max_pseudo_box_num:
                        max_pseudo_box_num = pseudo_box.shape[0]

                max_box_num = batch_dict['gt_boxes'].shape[1]
                #labeldense = torch.ones(1,max_box_num,1)
                #unlabeldense = torch.zeros(1,max_box_num,1)
                #densescores = torch.cat([labeldense,unlabeldense], dim=0)

                # assert max_box_num >= max_pseudo_box_num
                ori_unlabeled_boxes = batch_dict['gt_boxes'][unlabeled_mask, ...]

                if max_box_num >= max_pseudo_box_num:
                    for i, pseudo_box in enumerate(pseudo_boxes):
                        diff = max_box_num - pseudo_box.shape[0]
                        if diff > 0:
                            pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                        batch_dict['gt_boxes'][unlabeled_mask[i]] = pseudo_box
                else:
                    ori_boxes = batch_dict['gt_boxes']
                    new_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, ori_boxes.shape[2]),
                                            device=ori_boxes.device)
                    for i, inds in enumerate(labeled_mask):
                        diff = max_pseudo_box_num - ori_boxes[inds].shape[0]
                        new_box = torch.cat([ori_boxes[inds], torch.zeros((diff, 8), device=ori_boxes[inds].device)], dim=0)
                        new_boxes[inds] = new_box
                    for i, pseudo_box in enumerate(pseudo_boxes):

                        diff = max_pseudo_box_num - pseudo_box.shape[0]
                        if diff > 0:
                            pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                        new_boxes[unlabeled_mask[i]] = pseudo_box
                    batch_dict['gt_boxes'] = new_boxes

                batch_dict['gt_boxes'][unlabeled_mask, ...] = random_flip_along_x_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['flip_x'][unlabeled_mask, ...]
                )

                batch_dict['gt_boxes'][unlabeled_mask, ...] = random_flip_along_y_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['flip_y'][unlabeled_mask, ...]
                )

                batch_dict['gt_boxes'][unlabeled_mask, ...] = global_rotation_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['rot_angle'][unlabeled_mask, ...]
                )

                batch_dict['gt_boxes'][unlabeled_mask, ...] = global_scaling_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['scale'][unlabeled_mask, ...]
                )

            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)

            disp_dict = {}
            loss_rpn_cls, loss_rpn_box, tb_dict = self.pv_rcnn.dense_head.get_loss(scalar=False)
            loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict, scalar=False)
            loss_rcnn_cls, loss_rcnn_box, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict, scalar=False)

            if not self.unlabeled_supervise_cls:
                loss_rpn_cls = loss_rpn_cls[labeled_mask, ...].sum()
            else:
                loss_rpn_cls = loss_rpn_cls[labeled_mask, ...].sum() + loss_rpn_cls[unlabeled_mask, ...].sum() * self.unlabeled_weight

            loss_rpn_box = loss_rpn_box[labeled_mask, ...].sum() + loss_rpn_box[unlabeled_mask, ...].sum() * self.unlabeled_weight
            loss_point = loss_point[labeled_mask, ...].sum()
            loss_rcnn_cls = loss_rcnn_cls[labeled_mask, ...].sum()

            if not self.unlabeled_supervise_refine:
                loss_rcnn_box = loss_rcnn_box[labeled_mask, ...].sum()
            else:
                loss_rcnn_box = loss_rcnn_box[labeled_mask, ...].sum() + loss_rcnn_box[unlabeled_mask, ...].sum() * self.unlabeled_weight

            loss = loss_rpn_cls + loss_rpn_box + loss_point + loss_rcnn_cls + loss_rcnn_box
            tb_dict_ = {}
            for key in tb_dict.keys():
                if 'loss' in key:
                    tb_dict_[key+"_labeled"] = tb_dict[key][labeled_mask, ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_mask, ...].sum()
                elif 'acc' in key:
                    tb_dict_[key+"_labeled"] = tb_dict[key][labeled_mask, ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_mask, ...].sum()
                elif 'point_pos_num' in key:
                    tb_dict_[key + "_labeled"] = tb_dict[key][labeled_mask, ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_mask, ...].sum()
                else:
                    tb_dict_[key] = tb_dict[key]


            tb_dict_['max_box_num'] = max_box_num
            tb_dict_['max_pseudo_box_num'] = max_pseudo_box_num
            tb_dict_['num'] = num
            #tb_dict_['scorenum'] = scorenum
            tb_dict_['semnum'] = semnum
            tb_dict_['car'] = self.sem_thresh[0]
            tb_dict_['ped'] = self.sem_thresh[1]
            tb_dict_['cyc'] = self.sem_thresh[2]

            ret_dict = {
                'loss': loss
            }
            self.update_semthresh()

            
            return ret_dict, tb_dict_, disp_dict

        else:
            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)

            pred_dicts, recall_dicts = self.pv_rcnn.post_processing(batch_dict)

            return pred_dicts, recall_dicts, {}
    
    def update_semthresh(self):
        #0.02 1000  0.01 600
        sem_thresh = 0.6
        m=self.step
        k = int(m/self.decay_step)
        #print(m)
        thresh = torch.tensor(sem_thresh - k/10) 
        thresh = torch.clamp(thresh, min=self.minthr)
        self.sem_thresh[0] = thresh
        self.sem_thresh[1] = thresh
        self.sem_thresh[2] = thresh 
        #print(self.sem_thresh)
    
    def update_semthresh_exp(self):
        sem_thresh=0.7
        m=self.step
        a = 0.4*np.exp(1)/(np.exp(1)-1)
        b = -1/(2500)
        c = 0.3*np.exp(1)-0.7
        m = m.cpu().numpy()
        thresh = a*np.exp(b*m)+c-0.04
        thresh = torch.tensor(thresh).cuda()
        thresh = torch.clamp(thresh, min=0.3, max=0.7)
        self.sem_thresh[0] = thresh
        self.sem_thresh[1] = thresh
        self.sem_thresh[2] = thresh
        print(self.sem_thresh) 



    def get_supervised_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    def update_global_step(self):
        self.global_step += 1
        self.step += 1
        alpha = 0.999
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        for ema_param, param in zip(self.pv_rcnn_ema.parameters(), self.pv_rcnn.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            new_key = 'pv_rcnn.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
            new_key = 'pv_rcnn_ema.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
            new_key = key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))
    
    
