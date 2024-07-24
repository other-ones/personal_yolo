export CUDA_VISIBLE_DEVICES=0,1,2
python -m torch.distributed.run --nproc_per_node 3 --master_port 2910 train.py \
--data data/docbank_c5.yaml \
--cfg models/yolov5m.yaml \
--name="docbank_ddp_1e3lr_pretrain" --img 1024 \
--batch-size 72 --epochs 50 \
--device 0,1,2 --optimizer AdamW


export CUDA_VISIBLE_DEVICES=0,1,2
python -m torch.distributed.run --nproc_per_node 3 --master_port 2910 train.py \
--data data/synthlayout_c5.yaml \
--cfg models/yolov5m.yaml \
--name="synthlayout_ddp" --img 1024 \
--batch-size 72 \
--device 0,1,2


export CUDA_VISIBLE_DEVICES=0,1,2
python -m torch.distributed.run --nproc_per_node 3 --master_port 2910 train.py \
--data data/doclaynet_c6.yaml \
--cfg models/yolov5m.yaml \
--name="docbank_ft_1e2lr" --img 1024 \
--batch-size 72 \
--device 0,1,2 \
--weights='runs/train/docbank_ddp_1e3lr_pretrain/weights/best.pt'



export CUDA_VISIBLE_DEVICES=0,1,2;
python -m torch.distributed.run --nproc_per_node 3 --master_port 2910 train.py \
--data data/doclaynet_c3.yaml \
--cfg models/yolov5l.yaml \
--name="yolov5l_c3" --img 1024 \
--batch-size 48 \
--device 0,1,2 \
--epochs 300  

