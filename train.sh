
export CUDA_VISIBLE_DEVICES=3;
python train.py --data data/docbank_c5.yaml --epochs 300 --weights '' --cfg models/yolov5m.yaml  --batch-size 25 \
--name="tmp" --img 1024 --noautoanchor --exist-ok


export CUDA_VISIBLE_DEVICES=3;
python train.py --data data/synthlayout_c5.yaml --epochs 300 --weights '' --cfg models/yolov5m.yaml  --batch-size 25 \
--name="tmp" --img 1024 






export CUDA_VISIBLE_DEVICES=3;
python train.py --data data/doclaynet_c8_aug.yaml --epochs 300 --weights '' --cfg models/yolov5m.yaml  --batch-size 25 \
--name="yolov5m_c8_res1025_augmented" --img 1024 


export CUDA_VISIBLE_DEVICES=2;
python train.py --data data/doclaynet_c8_aug.yaml --epochs 300 --weights '' --cfg models/yolov5m.yaml  --batch-size 30 \
--name="tmp" --img 1024 


export CUDA_VISIBLE_DEVICES=3;
python train.py --data data/doclaynet_c6_sampled.yaml --epochs 300 --weights '' --cfg models/yolov5m_gray.yaml  --batch-size 15 \
--name="tmp" --img 1024 --num_channels=1

export CUDA_VISIBLE_DEVICES=3;
python train.py --data data/doclaynet_c6_sampled.yaml --epochs 300 --weights '' --cfg models/yolov5m.yaml  --batch-size 15 \
--name="tmp" --img 1024 --num_channels=3


export CUDA_VISIBLE_DEVICES=2;
python train.py --data data/doclaynet_c6.yaml --epochs 300 --weights '' --cfg models/yolov5l.yaml  --batch-size 48 \
--name="yolov5l_c6_fixed_cont" \
--resume='runs/train/yolov5l_c6_fixed2_backup/weights/last.pt'



export CUDA_VISIBLE_DEVICES=3;
python train.py --data data/doclaynet_c6_aug.yaml --epochs 300 --weights '' --cfg models/yolov5m.yaml  --batch-size 25 \
--name="tmp" --img 1024 


export CUDA_VISIBLE_DEVICES=2;
python train.py --data data/doclaynet_c3.yaml --epochs 300 --weights '' --cfg models/yolov5l.yaml  --batch-size 48 \
--name="yolov5l_c3" --img 1024
