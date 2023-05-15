import argparse
import os

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils, box_utils
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/voxel_rcnn_ssl.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--out_dir', type=str, default='../output/test')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Test-------------------------')

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        logger.info(f'Total samples: \t{len(test_loader)}')
        for idx, data_dict in enumerate(test_loader):
            logger.info(f'Tested sample index: \t{idx}')
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            name = str(data_dict['frame_id'][0])
            txt = []
            for i in range(pred_dicts[0]['pred_labels'].shape[0]):
                pred = pred_dicts[0]
                cls_type = np.array(cfg.CLASS_NAMES)[pred['pred_labels'][i].cpu().numpy() - 1]
                trucation = 0
                occlusion = 0
                pred_boxes = pred['pred_boxes'][[i]].cpu().numpy()
                calib = data_dict['calib'][0]
                image_shape = data_dict['image_shape'][0].cpu().numpy()
                pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
                pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                    pred_boxes_camera, calib, image_shape=image_shape
                )
                alpha = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
                bbox = pred_boxes_img
                # camera l h w
                # lidar l w h
                # output h w l
                dimensions = np.flip(pred_boxes[:, 3:6], -1)
                location = pred_boxes[:, 0:3]
                rotation_y = pred_boxes_camera[:, 6]
                confidence = pred['pred_scores'][i].cpu().numpy()
                result = np.concatenate((bbox, dimensions, location), -1)[0]
                result = [cls_type, trucation, occlusion, alpha[0]] + result.tolist() + [float(rotation_y), float(confidence)]
                result = ' '.join(map(str, result))
                if(confidence<0.0):
                    continue
                txt.append(result)

            dir = os.path.join(args.out_dir, cfg.MODEL.NAME)

            if not os.path.exists(dir):
                os.makedirs(dir)
            with open(os.path.join(dir, name + '.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(map(str, txt)))
                f.close()

    logger.info('Test done.')


if __name__ == '__main__':
    common_utils.set_random_seed(666)
    main()
