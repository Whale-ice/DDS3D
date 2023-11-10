# DDS3D
Dense Pseudo-Labels with Dynamic Threshold for Semi-Supervised 3D Object Detection(DDS3D).

You can find the paper at https://arxiv.org/abs/2303.05079.

This is the repository for DDS3D(ICRA2023).
In this repository, we provide DDS3D implementation (with pytorch) based on [PV-RCNN](https://github.com/open-mmlab/OpenPCDet) and [3DIoUMatch](https://github.com/THU17cyz/3DIoUMatch-PVRCNN).

i find some problems so this repo need to modify, i will fix them soon, please choose v2.0 branch. 
## Installation

Please refer to the origin [README.md](./README_OpenPCDet.md) for installation and usage of OpenPCDet.

## Data Preparation and Training

Please follow [3DIoUMatch-PVRCNN](https://github.com/THU17cyz/3DIoUMatch-PVRCNN)

#### Pre-training
For example

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_pretrain.sh  4 --cfg_file cfgs/kitti_models/pvrcnn.yaml  --extra_tag split_0.20_1 --split train_0.20_1 --ckpt_save_interval 4 --repeat 10 --dbinfos kitti_dbinfos_train_0.20_1_742.pkl
```

#### semi-training
For example
```bash
CUDA_VISIBLE_DEVICES=2,3 bash scripts/dist_train.sh 2 --cfg_file cfgs/kitti_models/pv_rcnn_ssl_db.yaml --split train_0.01_3 --extra_tag split_0.01_3 --ckpt_save_interval 2 --pretrained_model ../output/kitti_models/pvrcnn/split_0.01_3/ckpt/checkpoint_epoch_80.pth  --repeat 5 --thresh '0.5,0.25,0.25' --sem_thresh '0.4,0.0,0.0' --dbinfos kitti_dbinfos_train_0.01_3_37.pkl
```
#### Bibtes
If this work is helpful for your research, please consider citing the following BibTeX entry.
```bash
@INPROCEEDINGS{10160489,
  author={Li, Jingyu and Liu, Zhe and Hou, Jinghua and Liang, Dingkang},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={DDS3D: Dense Pseudo-Labels with Dynamic Threshold for Semi-Supervised 3D Object Detection}, 
  year={2023},
  volume={},
  number={},
  pages={9245-9252},
  doi={10.1109/ICRA48891.2023.10160489}}
```
