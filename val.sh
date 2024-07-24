
export CUDA_VISIBLE_DEVICES=1;
python val.py --data data/doclaynet_c6.yaml    --batch-size 80 \
--name="docbank_ddp_ft_doclaynet_c6_val" \
--weights='runs/train/docbank_ddp_ft_adamw/weights/best.pt' --img 1024 \
--task val --save-json



export CUDA_VISIBLE_DEVICES=1;
python val.py --data data/doclaynet_c6.yaml    --batch-size 80 \
--name="synthlayout_ddp_ft_doclaynet_c6_val" \
--weights='runs/train/synthlayout_ddp_ft/weights/best.pt' --img 1024 \
--task val --save-json

export CUDA_VISIBLE_DEVICES=1;
python val.py --data data/doclaynet_c6.yaml    --batch-size 80 \
--name="yolov5m_c6_res1025_noaug_doclaynet_c6_val" \
--weights='runs/train/yolov5m_c6_res1025_noaug/weights/best.pt' --img 1024 \
--task val --save-json

