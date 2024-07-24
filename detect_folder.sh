export CUDA_VISIBLE_DEVICES=6;
python detect.py --weights /home/twkim/project/yolov5/runs/train/exp8/weights/last.pt \
--img 640 --conf 0.25 --source data/images/doc_images/c02754ccc4e756f0c78cb44de9ef6049f181ad460d53a09744b26a9fe51237f2.png


export CUDA_VISIBLE_DEVICES=2;
python detect.py --weights /home/twkim/project/yolov5_qlab07/runs/train/yolov5m_c6_re/weights/best.pt \
--img 640 --conf 0.25 --source /data/twkim/doc_layout/raw/hp_layout_samples --save-txt