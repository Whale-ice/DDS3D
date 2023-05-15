#CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4  --cfg_file cfgs/kitti_models/pv_rcnn_ssl_dl.yaml  --split train_0.02_1 --extra_tag train_0.02_1_p  --pretrained_model /home/jingyu/competition/3DIoUMatch-PVRCNN/output/kitti_models/pv_rcnn/split_0.02_1_80/ckpt/checkpoint_epoch_72.pth  --repeat 5 --thresh '0.5,0.25,0.25'  --sem_thresh '0.4,0.4,0.4' --dbinfos kitti_dbinfos_train_0.02_1_74.pkl --epochs 60 --max_ckpt_save_num 20 
#CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4  --cfg_file cfgs/kitti_models/pv_rcnn_ssl_dl.yaml  --split train_0.02_2 --extra_tag train_0.02_2_p  --pretrained_model /home/jingyu/competition/3DIoUMatch-PVRCNN/output/kitti_models/pv_rcnn/split_0.02_2/ckpt/checkpoint_epoch_88.pth  --repeat 5 --thresh '0.5,0.25,0.25'  --sem_thresh '0.4,0.4,0.4' --dbinfos kitti_dbinfos_train_0.02_2_74.pkl --epochs 60  --max_ckpt_save_num 20
#CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4  --cfg_file cfgs/kitti_models/pv_rcnn_ssl_dl.yaml  --split train_0.02_3 --extra_tag train_0.02_3_p  --pretrained_model /home/jingyu/competition/3DIoUMatch-PVRCNN/output/kitti_models/pv_rcnn/split_0.02_3/ckpt/checkpoint_epoch_91.pth  --repeat 5 --thresh '0.5,0.25,0.25'  --sem_thresh '0.4,0.4,0.4' --dbinfos kitti_dbinfos_train_0.02_3_74.pkl --epochs 60  --max_ckpt_save_num 20
#CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4  --cfg_file cfgs/kitti_models/pv_rcnn_ssl_db.yaml  --split train_0.02_1 --extra_tag train_0.02_1_sem100_conlr  --pretrained_model /home/jingyu/competition/3DIoUMatch-PVRCNN/output/kitti_models/pv_rcnn/split_0.02_1_80/ckpt/checkpoint_epoch_72.pth  --repeat 5 --thresh '0.0,0.0,0.0'  --sem_thresh '0.6,0.6,0.6' --dbinfos kitti_dbinfos_train_0.02_1_74.pkl --epochs 100 --max_ckpt_save_num 20 
#sleep 200m
#CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4  --cfg_file cfgs/kitti_models/pv_rcnn_ssl_db.yaml  --split train_0.02_2 --extra_tag train_0.02_2_sem100_conlr  --pretrained_model /home/jingyu/competition/3DIoUMatch-PVRCNN/output/kitti_models/pv_rcnn/split_0.02_2/ckpt/checkpoint_epoch_88.pth  --repeat 5 --thresh '0.0,0.0,0.0'  --sem_thresh '0.6,0.6,0.6' --dbinfos kitti_dbinfos_train_0.02_2_74.pkl --epochs 100  --max_ckpt_save_num 20
#sleep 200m
#CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4  --cfg_file cfgs/kitti_models/pv_rcnn_ssl_db.yaml  --split train_0.02_3 --extra_tag train_0.02_3_sem100_conlr  --pretrained_model /home/jingyu/competition/3DIoUMatch-PVRCNN/output/kitti_models/pv_rcnn/split_0.02_3/ckpt/checkpoint_epoch_91.pth  --repeat 5 --thresh '0.0,0.0,0.0'  --sem_thresh '0.6,0.6,0.6' --dbinfos kitti_dbinfos_train_0.02_3_74.pkl --epochs 100  --max_ckpt_save_num 20
#CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4  --cfg_file cfgs/kitti_models/pv_rcnn_ssl_db.yaml  --ckpt ../output/kitti_models/pv_rcnn_ssl_db/train_0.02_3_sem/ckpt/ --extra_tag train_0.02_3_sem --eval_all
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4  --cfg_file cfgs/kitti_models/pv_rcnn_ssl_dbv4.yaml  --split train_0.02_2 --extra_tag train_0.02_2_100  --pretrained_model /home/jingyu/competition/3DIoUMatch-PVRCNN/output/kitti_models/pv_rcnn/split_0.02_2/ckpt/checkpoint_epoch_88.pth  --repeat 5 --thresh '0.0,0.0,0.0'  --sem_thresh '0.6,0.6,0.6' --dbinfos kitti_dbinfos_train_0.02_2_74.pkl --epochs 100  --max_ckpt_save_num 20
sleep 200m
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4  --cfg_file cfgs/kitti_models/pv_rcnn_ssl_dbv3.yaml  --split train_0.02_2 --extra_tag train_0.02_2  --pretrained_model /home/jingyu/competition/3DIoUMatch-PVRCNN/output/kitti_models/pv_rcnn/split_0.02_2/ckpt/checkpoint_epoch_88.pth  --repeat 5 --thresh '0.0,0.0,0.0'  --sem_thresh '0.6,0.6,0.6' --dbinfos kitti_dbinfos_train_0.02_2_74.pkl --epochs 60  --max_ckpt_save_num 20
sleep 140m
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4  --cfg_file cfgs/kitti_models/pv_rcnn_ssl_dbv3.yaml  --split train_0.02_2 --extra_tag train_0.02_2_100  --pretrained_model /home/jingyu/competition/3DIoUMatch-PVRCNN/output/kitti_models/pv_rcnn/split_0.02_2/ckpt/checkpoint_epoch_88.pth  --repeat 5 --thresh '0.0,0.0,0.0'  --sem_thresh '0.6,0.6,0.6' --dbinfos kitti_dbinfos_train_0.02_2_74.pkl --epochs 100  --max_ckpt_save_num 20 