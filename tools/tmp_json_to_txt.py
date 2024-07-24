from PIL import Image
import json
import os
import cv2
import numpy as np

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
        if colors:
            for box,color in zip(boxes,colors):
                box=np.array(box).reshape(-1,2)
                point1=box[0]
                point2=box[1]
                image=cv2.rectangle(image,tuple(point1),tuple(point2),color=color,thickness=thickness)
        else:
            for box in boxes:
                box=np.array(box).reshape(-1,2)
                point1=box[0]
                point2=box[1]
                image=cv2.rectangle(image,tuple(point1),tuple(point2),color=(0,255,0),thickness=thickness)
    image=Image.fromarray(image)
    return image
mode='test'
dst_root='/home/twkim/project/yolov5_qlab07/sampled1K/yolo_preds'
img_root='/data/twkim/doc_layout/raw/doclaynet/yolo/images/{}'.format(mode)
os.makedirs(dst_root,exist_ok=True)
data=json.load(open('/home/twkim/project/yolov5_qlab07/runs/val/docbank_ddp_ft_doclaynet_c6_{}_sampled/best_predictions.json'.format(mode)))
per_id_preds={}
for item in data:
    image_id=item['image_id']
    cat=item['category_id']
    bbox=item['bbox']
    score=item['score']
    if score<0.001:
        continue
    if image_id not in per_id_preds:
        per_id_preds[image_id]=[(cat,bbox)]
    else:
        per_id_preds[image_id].append((cat,bbox))
keys=list(per_id_preds.keys())
np.random.shuffle(keys)
for key in keys:
    dst_path=os.path.join(dst_root,key+'.txt')
    dst_file=open(dst_path,'w')
    img_path=os.path.join(img_root,key+'.png')
    img=Image.open(img_path)
    preds=per_id_preds[key]
    boxes=[]
    for pred in preds:
        cat,bbox=pred
        center_x,center_y,width,height=bbox
        x0,y0,width,height=bbox
        x1=x0+width
        y1=y0+height
        dst_file.write('{} {} {} {} {}\n'.format(cat,x0,y0,x1,y1))
        # box=np.array([x0,y0,x1,y1]).astype(np.int32)
        # boxes.append(box)
    # drawn=visualize_box(img,boxes)
    # drawn.save('drawn.jpg')
    