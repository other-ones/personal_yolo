export CUDA_VISIBLE_DEVICES=1;
python detect.py --weights /home/twkim/project/yolov5_qlab07/runs/train/yolov5m_c6_fixed_batch80/weights/best.pt \
--img 720 --conf 0.25 --source 1000layout_samples_to_purdue --name yolov5m_c6_fixed_batch80_720_eval


export CUDA_VISIBLE_DEVICES=3;
python detect.py --weights /home/twkim/project/yolov5_qlab07/runs/train/yolov5m_c8_res1025_augmented/weights/best.pt \
--img 1025 --conf 0.15 --source 1000layout_samples_to_purdue --name yolov5m_c8_res1025_augmented_eval_th15

export CUDA_VISIBLE_DEVICES=1;
python detect.py --weights /home/twkim/project/yolov5_qlab07/runs/train/yolov5m_c6_res1025_augmented/weights/best.pt \
--img 1025 --conf 0.25 --source 1000layout_samples_to_purdue --name yolov5m_c6_res1025_augmented_th025_res1025




export CUDA_VISIBLE_DEVICES=1;
python detect.py --weights runs/train/yolov5m_c6_res1025_noaug/weights/best.pt \
--img 1025 --conf 0.25 --source 1000layout_samples_to_purdue --name yolov5m_c6_res1025_noaug_th025_res1025

export CUDA_VISIBLE_DEVICES=2;
python val.py --data data/doclaynet_c6.yaml    --batch-size 80 \
--name="yolov5m_c6_res1025_noaug_1025eval" \
--weights='runs/train/yolov5m_c6_res1025_noaug/weights/best.pt' --img 1025 \
--save-json
