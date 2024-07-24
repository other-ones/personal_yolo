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
            image=cv2.rectangle(image,tuple(point1),tuple(point2),color=(0,255,0),thickness=1)
    image=Image.fromarray(image)
    return image



# # 1. Pyt n6
# result_data=json.load(open('/home/twkim/project/doclay_det/c5_results/coco_instances_results.json'))
# pred_items=[]
# for item in result_data:
#     img_id=item['image_id']
#     if img_id!=(target_id):
#         continue
#     pred_items.append(item)
gt_data=json.load(open('/data/twkim/doc_layout/raw/doclaynet/COCO/train_c6.json'))
anns=gt_data['annotations']
target_id=np.random.choice(np.arange(200))
# 2. Ground Truth
gt_items=[]
for item in anns[:]:
    img_id=item['image_id']
    if img_id!=(target_id):
        continue
    gt_items.append(item)
img_data=gt_data['images']
for item in img_data:
    img_id=item['id']
    if img_id!=(target_id):
        continue
    file_name=item['file_name']
    break


cat_map={
    1:"text",
    2:"table",
    3:"figure",
    4:"formula",
    5:"footer",
    6:"header",
}
img_root='/data/twkim/doc_layout/raw/doclaynet/PNG/'
img_path=os.path.join(img_root,file_name)
print(img_path)
img=Image.open(img_path).convert('RGB')
# pred_boxes=[]
# for item in pred_items:
#     box=np.array(item['bbox']).astype(np.int32)
#     pred_boxes.append(box)





gt_boxes=[]
cat_names=[]
colors=[]
np.random.shuffle(gt_items)
for item in gt_items[:]:
    catid=item['category_id']
    cat_name=cat_map[catid]
    if cat_name=='text':
        color=(255,0,0)
    elif cat_name=='figure':
        color=(0,255,0)
    elif cat_name=='formula':
        color=(0,0,255)
    elif cat_name=='footer':
        color=(125,0,255)
    elif cat_name=='table':
        color=(0,225,125)
    box=np.array(item['bbox']).astype(np.int32)
    x,y,w,h=box
    x0,y0,x1,y1=x,y,x+w,y+h
    box=[x0,y0,x1,y1]
    gt_boxes.append(box)
    cat_names.append(cat_name)
    colors.append(color)
    
    
    

drawn_gt=visualize_box(img,gt_boxes,chars=cat_names,colors=colors)
drawn_gt.save('drawn_gt.jpg')


# drawn_pred=visualize_box(img,pred_boxes)
# drawn_pred.save('drawn_pred.jpg')