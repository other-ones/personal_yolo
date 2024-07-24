import cv2
import os
import numpy as np
import shutil


mode='val'
cfg='c6'
root='/data/twkim/doc_layout/raw/doclaynet/yolo/'
img_root=os.path.join(root,'images2500/{}_{}'.format(mode,cfg))
label_root=os.path.join(root,'images2500/{}_{}'.format(mode,cfg))

dst_root='/home/twkim/project/yolov5_qlab07/sampled1K/{}'.format(mode)
dst_img_root=os.path.join(dst_root,'images')
dst_label_root=os.path.join(dst_root,'labels')
os.makedirs(dst_img_root,exist_ok=True)
os.makedirs(dst_label_root,exist_ok=True)

val_list_path=os.path.join(root,'{}_{}.txt'.format(mode,cfg))
val_lines=open(val_list_path).readlines()
np.random.shuffle(val_lines)
count=0
for line in val_lines:
    line=line.strip()
    label_line=line.replace('images','labels').replace(".png",".txt")
    src_img_path=os.path.join(root,line)
    src_label_path=os.path.join(root,label_line)
    if not(os.path.exists(src_img_path) and os.path.exists(src_label_path)):
        continue
    dst_img_path=os.path.join(dst_root,line.replace('{}_{}/'.format(mode,cfg),'').replace('.png','.jpeg'))
    dst_label_path=os.path.join(dst_root,label_line.replace('{}_{}/'.format(mode,cfg),''))
    img=cv2.imread(src_img_path)
    cv2.imwrite(dst_img_path,img)
    shutil.copy(src_label_path,dst_label_path)
    count+=1
    if count==1000:
        break



    
