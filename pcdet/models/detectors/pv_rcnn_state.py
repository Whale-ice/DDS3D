from .detector3d_template import Detector3DTemplate


class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts, {}

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
    
    def vis_post_processing(self, batch_dict, no_recall_dict=False, override_thresh=None, no_nms=False):
        import torch
        import pdb
        pdb.set_trace()
        sem_thresh= [0.4,0.4,0.4]
        pred_dicts = []
        box_preds = batch_dict['batch_box_preds'][0]
        cls_preds = batch_dict['batch_cls_preds'][0]
        cls_preds = torch.sigmoid(cls_preds)
        cls_preds, label_preds = torch.max(cls_preds, dim=-1)
        label_preds = batch_dict['roi_labels'][0]
        sem_scores = batch_dict['roi_scores'][0]
        final_sem_scores = torch.sigmoid(sem_scores)
        record_dicts = {
            'pred_boxes': box_preds,
            'pred_scores': cls_preds,
            'pred_labels': label_preds,
            'pred_sem_scores': final_sem_scores
        }
        pseudo_score = record_dicts['pred_scores']
        pseudo_box = record_dicts['pred_boxes']
        pseudo_label = record_dicts['pred_labels']
        pseudo_sem_score = record_dicts['pred_sem_scores']
        conf_thresh = torch.tensor(sem_thresh,device=pseudo_label.device)
        conf_thresh = conf_thresh.unsqueeze(0).repeat(len(pseudo_label),1)
        conf_thresh = conf_thresh.gather(dim=1,index=(pseudo_label-1).unsqueeze(-1))

        #valid_inds = pseudo_score > conf_thresh.squeeze()
        #scorenum = valid_inds.sum
        valid_inds = pseudo_sem_score > conf_thresh.squeeze()
        num = valid_inds.sum()
    
        
        pseudo_sem_score = pseudo_sem_score[valid_inds]
        batch_dict['densescore'] = pseudo_sem_score
        pseudo_box = pseudo_box[valid_inds]
        pseudo_label = pseudo_label[valid_inds]
        pseudo_score = pseudo_score[valid_inds]
        
        record_dicts['pred_boxes'] = pseudo_box
        record_dicts['pred_scores'] = pseudo_score
        record_dicts['pred_labels'] = pseudo_label
        record_dicts['pred_sem_scores'] = pseudo_sem_score


        pred_dicts = []
        pred_dicts.append(record_dicts)
        return pred_dicts, None
