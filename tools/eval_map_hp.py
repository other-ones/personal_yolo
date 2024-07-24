import cv2
from PIL import Image
import os
import numpy as np
from mean_average_precision import MetricBuilder


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

def parse_pred(fpath,binary=False,target_cls=None):
    confidence=1
    lines=open(fpath).readlines()
    header_lines=lines[:5]
    num_txt_blocks=int(header_lines[0].split(': ')[-1]) #10
    num_txt_lines=int(header_lines[1].split(': ')[-1]) #10
    num_images=int(header_lines[2].split(': ')[-1]) #2
    num_graphic=int(header_lines[3].split(': ')[-1]) #0
    num_tables=int(header_lines[4].split(': ')[-1]) #0
    data_lines = lines[5:]
    text_block_data=data_lines[1:1+num_txt_blocks]
    text_line_data=data_lines[2+num_txt_blocks:2+num_txt_blocks+num_txt_lines]
    image_data=data_lines[3+num_txt_blocks+num_txt_lines:3+num_txt_blocks+num_txt_lines+num_images]
    graphic_data=data_lines[4+num_txt_blocks+num_txt_lines+num_images:4+num_txt_blocks+num_txt_lines+num_images+num_graphic]
    table_data=data_lines[5+num_txt_blocks+num_txt_lines+num_images+num_graphic:5+num_txt_blocks+num_txt_lines+num_images+num_graphic+num_tables]
    cat_list=[]
    dst_list=[]
    for line in text_line_data: #text
        line=line.strip()
        splits=line.split()
        bbox=np.array(splits[1:]).astype(np.int32)
        y0, y1, x0, x1 = bbox
        cat=0
        if target_cls is not None and target_cls !=cat:
            continue
        if binary:
            cat=0
        cat_list.append(cat)
        bbox=[x0,y0,x1,y1]
        # dst_item=[x0,y0,x1,y1,cat,confidence]
        dst_item=[x0,y0,x1,y1]
        dst_list.append(dst_item)

    for line in image_data: #figure
        line=line.strip()
        splits=line.split()
        bbox=np.array(splits[1:]).astype(np.int32)
        y0, y1, x0, x1 = bbox
        bbox=[x0,y0,x1,y1]
        cat=2
        if target_cls is not None and target_cls !=cat:
            continue
        if binary:
            cat=0
        # dst_item=[x0,y0,x1,y1,cat,confidence]
        dst_item=[x0,y0,x1,y1]
        dst_list.append(dst_item)


    for line in graphic_data: #graphic
        line=line.strip()
        splits=line.split()
        bbox=np.array(splits[1:]).astype(np.int32)
        cat=1
        if target_cls is not None and target_cls !=cat:
            continue
        if binary:
            cat=0
        y0, y1, x0, x1 = bbox
        # dst_item=[x0,y0,x1,y1,cat,confidence]
        dst_item=[x0,y0,x1,y1]
        dst_list.append(dst_item)


    for line in table_data: # table
        line=line.strip()
        splits=line.split()
        bbox_coords=np.array(splits[1:]).astype(np.int32)
        cat=1
        if target_cls is not None and target_cls !=cat:
            continue
        if binary:
            cat=0
        if len(bbox_coords)!=4:
            assert (len(bbox)%4)==0
            num_boxes=len(bbox)//4
            for bidx in range(num_boxes):
                bbox=bbox_coords[bidx:(bidx+1)*4]
                y0, y1, x0, x1 = bbox
                bbox = [x0, y0, x1, y1]
        else:
            y0, y1, x0, x1 = bbox
            bbox = [x0, y0, x1, y1]
        # dst_item=[x0,y0,x1,y1,cat,confidence]
        dst_item=[x0,y0,x1,y1]
        dst_list.append(dst_item)
    # img_path=os.path.join(fpath.replace('hp_preds','images').replace('.jpeg_LAYOUT_INFO.txt','.jpeg'))
    # img=Image.open(img_path)
    # drawn=visualize_box(img,dst_list)
    # drawn.save('hp_drawn.jpg')
    # exit()
    # exit()
    return dst_list
def parse_gt(fpath,binary=False,target_cls=None):
    lines=open(fpath).readlines()
    # 0a3bc6f54adeedfb7b60678a83a89bb0f4d0135dc26a7a8d89a3ae2e3ccbf98d.jpeg_LAYOUT_INFO
    dst_list=[]
    for line in lines:
        line=line.strip()
        splits=line.split()
        center_x,center_y,width,height=np.array(splits[1:]).astype(np.float32)*1025
        cat=int(splits[0])
        if (target_cls is not None) and target_cls !=cat:
            continue
        if binary:
            cat=0
        x0=int(center_x-width/2)
        x1=int(x0+width)
        y0=int(center_y-height/2)
        y1=int(y0+height)
        # dst_item=[x0,x1,y0,y1,cat,0,0]
        dst_item=[x0,y0,x1,y1]
        dst_list.append(dst_item)
    # img_path=os.path.join(fpath.replace('labels','images').replace('.txt','.jpeg'))
    # img=Image.open(img_path)
    # drawn=visualize_box(img,dst_list)
    # drawn.save('gt_drawn.jpg')
    # exit()
    return dst_list

import numpy as np

def compute_iou(box1, box2):
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p = box2[0], box2[1], box2[2]
    y2_p = box2[3]
    
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    return iou

def compute_ap(recalls, precisions):
    """Compute the average precision, given the recall and precision curves."""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

def mean_average_precision(ground_truths, predictions, iou_thresholds):
    """Compute the Mean Average Precision (mAP) at different IoU thresholds."""
    aps = {iou_threshold: [] for iou_threshold in iou_thresholds}
    for iou_threshold in iou_thresholds:
        for gt_boxes, pred_boxes in zip(ground_truths, predictions):
            if len(pred_boxes) == 0:
                aps[iou_threshold].append(0)
                continue
            
            gt_matched = np.zeros(len(gt_boxes))
            pred_matched = np.zeros(len(pred_boxes))
            
            ious = np.zeros((len(pred_boxes), len(gt_boxes)))
            for i, pred_box in enumerate(pred_boxes):
                for j, gt_box in enumerate(gt_boxes):
                    ious[i, j] = compute_iou(pred_box, gt_box)
            
            tp = np.zeros(len(pred_boxes))
            fp = np.zeros(len(pred_boxes))
            
            for i in range(len(pred_boxes)):
                max_iou_idx = np.argmax(ious[i])
                if ious[i, max_iou_idx] >= iou_threshold:
                    if not gt_matched[max_iou_idx]:
                        tp[i] = 1
                        gt_matched[max_iou_idx] = 1
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1
            
            tp = np.cumsum(tp)
            fp = np.cumsum(fp)
            
            recalls = tp / len(gt_boxes)
            precisions = tp / (tp + fp)
            
            ap = compute_ap(recalls, precisions)
            aps[iou_threshold].append(ap)
    
    mAPs = {iou_threshold: np.mean(aps[iou_threshold]) for iou_threshold in iou_thresholds}
    return mAPs















mode='val'
pred_root='/home/twkim/project/yolov5_qlab07/sampled1K/{}/hp_preds_{}'.format(mode,mode)
label_root='/home/twkim/project/yolov5_qlab07/sampled1K/{}/labels_{}'.format(mode,mode)
flist=os.listdir(label_root)
ground_truths=[]
predictions=[]
target_cls=0
# np.random.shuffle(flist)
for ff in flist:
    pred_path=os.path.join(pred_root,ff.replace('.txt','.jpeg_LAYOUT_INFO.txt'))
    gt_path=os.path.join(label_root,ff)
    if (not os.path.exists(pred_path)) or (not os.path.exists(gt_path)):
        continue
    gt=parse_gt(gt_path,binary=True,target_cls=target_cls)
    if not len(gt):
        continue
    pred=parse_pred(pred_path,binary=True,target_cls=target_cls)
    ground_truths.append(gt)
    predictions.append(pred)


iou_thresholds = np.arange(0.5, 1.0, 0.05)
mAPs = mean_average_precision(ground_truths, predictions, iou_thresholds)
avg=0
print('Target Class:\t{}'.format(target_cls))
for key in mAPs:
    print('{}\t{}'.format(key,mAPs[key]))
    avg+=mAPs[key]
print('avg\t{}'.format(avg/len(mAPs)))

