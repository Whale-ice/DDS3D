# DDS3D
Dense Pseudo-Labels with Dynamic Threshold for Semi-Supervised 3D Object Detection(DDS3D).

You can find the paper at https://arxiv.org/abs/2303.05079.

This is the repository for DDS3D(ICRA2023).
In this repository, we provide DDS3D implementation (with pytorch) based on PV-RCNN and 3DIoUMatch.


## Installation

Please refer to the origin [README.md](./README_OpenPCDet.md) for installation and usage of OpenPCDet.

## Data Preparation and Training

#### Data preparation

Please first generate the data splits or use the data splits we provide.

```bash
cd data/kitti/ImageSets
python split.py <label_ratio> <split_num>
cd ../../..
```

For example:

```bash
cd data/kitti/ImageSets
python split.py 0.01 4
cd ../../..
```

Then generate the `infos` and `dbinfos`, and rename `kitti_dbinfos_train_3712.pkl`.

```
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \
tools/cfgs/dataset_configs/kitti_dataset.yaml
mv data/kitti/kitti_dbinfos_train_3712.pkl data/kitti/kitti_dbinfos_train.pkl
```

Then generate the new `gt_database` based on the data split.

```bash
python -m pcdet.datasets.kitti.kitti_dataset create_part_dbinfos \
tools/cfgs/dataset_configs/kitti_dataset.yaml <split_name_Without_txt>
```

For example:

```bash
python -m pcdet.datasets.kitti.kitti_dataset create_part_dbinfos \
tools/cfgs/dataset_configs/kitti_dataset.yaml train_0.01_1
```

#### Pre-training

```bash
GPUS_PER_NODE=<num_gpus> sh scripts/slurm_pretrain.sh <partition> \
<job_name> <num_gpus> --cfg_file ./cfgs/kitti_models/pv_rcnn.yaml \
--split <split_name_without_txt> --extra_tag <log_folder_name> \
--ckpt_save_interval <ckpt_save_interval> \
--repeat <number_of_traverses_of_dataset_in_one_epoch> \
--dbinfos <pkl_name_of_dbinfos>
```

For example:

```bash
GPUS_PER_NODE=8 sh scripts/slurm_pretrain.sh p1 pretrain_0.01_1 8 \
--cfg_file ./cfgs/kitti_models/pv_rcnn.yaml --split train_0.01_1 \
--extra_tag split_0.01_1 --ckpt_save_interval 4 --repeat 10 \
--dbinfos kitti_dbinfos_train_0.01_1_37.pkl
```

#### Training

```bash
GPUS_PER_NODE=<num_gpus> sh scripts/slurm_train.sh <partition> \
<job_name> <num_gpus> --cfg_file ./cfgs/kitti_models/pv_rcnn_ssl_60.yaml \
--split <split_name_without_txt> --extra_tag <log_folder_name> \
--ckpt_save_interval <ckpt_save_interval> --pretrain_model <path_to_pretrain_model> \
--repeat <number_of_traverses_of_dataset_in_one_epoch> --thresh <iou_thresh> \
--sem_thresh <sem_cls_thresh> --dbinfos <pkl_name_of_dbinfos>
```

For example:

```bash
GPUS_PER_NODE=8 sh scripts/slurm_train.sh p1 train_0.01_1 8 \
--cfg_file ./cfgs/kitti_models/pv_rcnn_ssl_60.yaml --split train_0.01_1 \
--extra_tag split_0.01_1 --ckpt_save_interval 2 \
--pretrained_model "../output/cfgs/kitti_models/pv_rcnn/split_0.01_1/ckpt/checkpoint_epoch_80.pth" \
--repeat 5 --thresh '0.5,0.25,0.25' --sem_thresh '0.4,0.0,0.0' \
--dbinfos kitti_dbinfos_train_0.01_1_37.pkl
```

Note: Currently only the first element of `sem_thresh` is used (class-agnostic). And the batch size per GPU card is currently hardcoded to be 1+1 (labeled+unlabeled).

## Acknowledgement

This codebase is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) (commit a7cf5368d9cbc3969b4613c9e61ba4dcaf217517).



CUDA_VISIBLE_DEVICES=2,3 bash scripts/dist_train.sh 2 --cfg_file cfgs/kitti_models/voxel_rcnn_ssl.yaml --split train_0.01_3 --extra_tag split_0.01_3 --ckpt_save_interval 2 --pretrained_model ../output/kitti_models/voxel_rcnn/split_0.01_3/ckpt/checkpoint_epoch_80.pth  --repeat 5 --thresh '0.5,0.25,0.25' --sem_thresh '0.4,0.0,0.0' --dbinfos kitti_dbinfos_train_0.01_3_37.pkl


预训练存在框回归不准，类别判断错误地问题，cyclist容易判断为pedestrain

CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh  4 --cfg_file cfgs/kitti_models/second.yaml  --extra_tag split_0.20_1 --split train_0.20_1 --ckpt_save_interval 4 --repeat 10 --dbinfos kitti_dbinfos_train_0.20_1_742.pkl
