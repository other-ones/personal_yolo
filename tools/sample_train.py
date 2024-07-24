import shutil
import numpy as np
import cv2
from PIL import Image

import os
import json
def visualize_box(image,boxes,chars=None,colors=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    image=np.array(image)
    if chars is not None:
        for box,char,color in zip(boxes,chars,colors):
            box=np.array(box)
            x0,y0,x1,y1=box
            box=(box).reshape(-1,2).astype(np.int32)
            point1=box[0]
            point2=box[1]
            image=cv2.rectangle(image,tuple(point1),tuple(point2),color=color,thickness=2)   
            image = cv2.putText(image, char, (x0,y1+15), font, fontScale, color, 2, cv2.LINE_AA)
    else:
        for box in boxes:
            box=np.array(box).reshape(-1,2)
            point1=box[0]
            point2=box[1]
            image=cv2.rectangle(image,tuple(point1),tuple(point2),color=(0,255,0),thickness=2)
    image=Image.fromarray(image)
    return image

cat_map={
    1:"text",
    2:"table",
    3:"figure",
    4:"formula",
    5:"footer",
    6:"header",
}
img_root='/data/twkim/doc_layout/raw/doclaynet/yolo/'
label_root='/data/twkim/doc_layout/raw/doclaynet/yolo/'
dst_dir='samples'
if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
os.makedirs(dst_dir,exist_ok=True)
lines=open('/data/twkim/doc_layout/raw/doclaynet/yolo/train_c6.txt').readlines()
np.random.shuffle(lines)
lines=lines[:100]
for line in lines:
    line=line.strip()
    fname=line.split('/')[-1]
    img_path=img_root+line
    label_path=label_root+line.replace('images/','labels/').replace('.png','.txt')
    if not os.path.exists(img_path):
        continue
    img=Image.open(img_path).convert('RGB')
    width,height=img.size
    print(width,height)
    label_lines=open(label_path).readlines()
    for ll in label_lines:
        ll=ll.strip()
        splits=ll.split()
        coords=np.array(splits[1:]).astype(np.float32)*width
        coords=coords.astype(np.int32)
        center_x,center_y,box_width,box_height=coords
        x0=center_x-(box_width/2)
        y0=center_y-(box_height/2)
        box=np.array([x0,y0,x0+box_width,y0+box_height]).astype(np.int32)
        img=visualize_box(img,[box])
    dst_path=os.path.join(dst_dir,fname)
    # shutil.copy(img_path,dst_path)
    img.save(dst_path)


