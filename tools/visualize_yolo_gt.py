import numpy as np
import cv2
from PIL import Image

import os
import json

def visualize_box(image,boxes,chars=None,colors=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    image=np.array(image)
    thickness=2
    if chars is not None:
        for box,char,color in zip(boxes,chars,colors):
            box=np.array(box)
            x0,y0,x1,y1=box
            box=(box).reshape(-1,2).astype(np.int32)
            point1=box[0]
            point2=box[1]
            if char not in ['column','row']:
                thickness=2
                image = cv2.putText(image, char, (x0,y1+15), font, fontScale, color, 1, cv2.LINE_AA)

                
            else:
                thickness=1
            image=cv2.rectangle(image,tuple(point1),tuple(point2),color=color,thickness=thickness)   
    else:
        for box,color in zip(boxes,colors):
            box=np.array(box).reshape(-1,2)
            point1=box[0]
            point2=box[1]
            image=cv2.rectangle(image,tuple(point1),tuple(point2),color=color,thickness=thickness)
    image=Image.fromarray(image)
    return image

image_root='/data/twkim/doc_layout/raw/docbank/yolo/images/'
label_root='/data/twkim/doc_layout/raw/docbank/yolo/labels/train_c5'
flist=os.listdir(label_root)
# ann={
  # "categories": [
  # {"id": 0,"name": "text","supercategory": ""},
  # {"id": 1,"name": "table","supercategory": ""},
  # {"id": 2,"name": "figure","supercategory": ""},
  # {"id": 3,"name": "formula","supercategory": ""},
  # {"id": 4,"name": "footer","supercategory": ""}]
  # }
np.random.shuffle(flist)
for ff in flist:
    label_path=os.path.join(label_root,ff)
    img_path=os.path.join(image_root,ff.replace('.txt','.jpg'))
    img=Image.open(img_path)
    img_w,img_h=img.size
    lines=open(label_path).readlines()
    boxes=[]
    colors=[]
    for line in lines:
        line=line.strip()
        splits=line.split()
        bbox=np.array(splits[1:]).astype(np.float32)
        bbox[[0,2]]=bbox[[0,2]]*img_w
        bbox[[1,3]]=bbox[[1,3]]*img_h
        center_x,center_y,box_w,box_h=bbox
        x0=center_x-box_w/2
        x1=x0+box_w
        y0=center_y-box_h/2
        y1=y0+box_h
        cat=int(splits[0])
        if cat==0:
            color=(0,255,165)
        elif cat==1:
            color=(255,0,0)
        elif cat==2:
            color=(255,165,0)
        else:
            color=(0,0,255)
        colors.append(color)
        boxes.append(np.array([x0,y0,x1,y1]).astype(np.int32))
    drawn=visualize_box(img,boxes,colors=colors)
    drawn.save('drawn.jpg')
    exit()
