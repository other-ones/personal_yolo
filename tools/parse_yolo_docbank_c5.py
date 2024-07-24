import cv2
from PIL import Image
import os
import numpy as np
import shutil
import json
ann={
"categories": [
{"id": 1,"name": "text","supercategory": ""},
{"id": 2,"name": "table","supercategory": ""},
{"id": 3,"name": "figure","supercategory": ""},
{"id": 4,"name": "formula","supercategory": ""},
{"id": 5,"name": "footer","supercategory": ""}]
}

conf='c5'
root='/data/twkim/doc_layout/raw/docbank'
dst_root='/data/twkim/doc_layout/raw/docbank/yolo'
dst_img_root=os.path.join(dst_root,'images')
dst_ann_root=os.path.join(dst_root,'labels')
os.makedirs(dst_img_root,exist_ok=True)
os.makedirs(dst_ann_root,exist_ok=True)

src_img_root=os.path.join(root,'yolo/images')
src_ann_root=os.path.join(root,'COCO')



val_boxes=[]
val_fname=None
for mode in ['train','val','test']:
  print('loading',mode)
  dst_img_mode_path=os.path.join(dst_img_root,mode)
  dst_ann_mode_path=os.path.join(dst_ann_root,mode+'_{}'.format(conf))
  if os.path.exists(dst_ann_mode_path):
    print('delete', dst_ann_mode_path)
    shutil.rmtree(dst_ann_mode_path)
  print('done')
  os.makedirs(dst_img_mode_path,exist_ok=True)
  os.makedirs(dst_ann_mode_path,exist_ok=True)
  src_ann_file=open(os.path.join(src_ann_root,'500K_{}.json'.format(mode)))
  data=json.load(src_ann_file)
  print(data.keys())
  print(data['categories'])
  # [{'id': 1, 'name': 'paragraph', 'supercategory': ''}, 
  # {'id': 10, 'name': 'table', 'supercategory': ''}, 
  # {'id': 5, 'name': 'figure', 'supercategory': ''}, 
  # {'id': 4, 'name': 'equation', 'supercategory': ''}, 
  # {'id': 11, 'name': 'footer', 'supercategory': ''}, 


  # {'id': 2, 'name': 'section', 'supercategory': ''}, 
  # {'id': 3, 'name': 'caption', 'supercategory': ''}, 
  # {'id': 6, 'name': 'date', 'supercategory': ''}, 
  # {'id': 7, 'name': 'abstract', 'supercategory': ''}, 
  # {'id': 8, 'name': 'author', 'supercategory': ''}, 
  # {'id': 9, 'name': 'title', 'supercategory': ''}, 
  # {'id': 12, 'name': 'reference', 'supercategory': ''}, 
  # {'id': 13, 'name': 'list', 'supercategory': ''}]
  # 0:text, 


  # ann={
  # "categories": [
  # {"id": 0,"name": "text","supercategory": ""},
  # {"id": 1,"name": "table","supercategory": ""},
  # {"id": 2,"name": "figure","supercategory": ""},
  # {"id": 3,"name": "formula","supercategory": ""},
  # {"id": 4,"name": "footer","supercategory": ""}]
  # }
  idmap={
    1:0,
    10:1,
    5:2,
    4:3,
    11:4,

    2:0,
    3:0,
    6:0,
    7:0,
    8:0,
    9:0,
    12:0,
    13:0,
  }
  img_data_list=data['images']
  ann_data=data['annotations']
  id2fname={}
  fname2id={}
  id2img={}
  print(data.keys(),'data.keys()')
  print(len(img_data_list),'len(img_data_list)')
  print(len(ann_data),'len(ann_data)')
  


  list_file=open(os.path.join(dst_root,'{}_{}.txt'.format(mode,conf)),'w')
  for img_data in img_data_list:
    img_id=img_data['id']
    fname=img_data['file_name']
    list_file.write('./images/{}/{}\n'.format(mode+'_{}'.format(conf),fname))
    fpath=os.path.join(src_img_root,fname)
    if not os.path.exists(fpath):
      continue
    id2fname[img_id]=fname
    fname2id[fname]=img_id
    id2img[img_id]=img_data


  # Parse Resolution data
  id2res={}
  res_path='res_docbank_{}.txt'.format(mode)
  if not os.path.exists(res_path):
    print('parse resolution',len(img_data_list),type(img_data_list))
    res_file=open('res_docbank_{}.txt'.format(mode),'w')
    res_file.write('fname\twidth\theight\n')
    for img_data in img_data_list:
      fname=img_data['file_name']
      img_width,img_height=img_data['width'],img_data['height']
      fpath=os.path.join(src_img_root,fname)
      if not os.path.exists(fpath):
        continue
      res_file.write('{}\t{}\t{}\n'.format(fname,img_width,img_height))
      img_id=fname2id[fname]
      id2res[img_id]=(img_width,img_height)
  else:
    res_file=open('res_docbank_{}.txt'.format(mode),'r')
    for line in res_file.readlines()[1:]:
      line=line.strip()
      splits=line.split('\t')
      fname=splits[0]
      img_width,img_height=np.array(splits[1:]).astype(np.int32)
      img_id=fname2id[fname]
      id2res[img_id]=(img_width,img_height)
  # Parse Resolution data


  


  for idx,ann in enumerate(ann_data):
    cat_id=ann['category_id']
    cat_new=idmap[cat_id]
    image_id=ann['image_id']
    if not image_id in id2res:
      continue
    img_w,img_h=id2res[image_id]
    bbox=ann['bbox']
    box_x,box_y,box_w,box_h=bbox
    center_x=box_x+(box_w/2)
    center_y=box_y+(box_h/2)
    new_bbox=[
              (center_x/img_w),
              (center_y/img_h),
              (box_w/img_w),
              (box_h/img_h)
    ]
    new_box=np.array(new_bbox)
    new_box=np.clip(new_bbox,0,1).tolist()
    if not image_id in id2fname:
      assert False
    fname=id2fname[image_id]
    src_image_path=os.path.join(src_img_root,fname)
    dst_image_path=os.path.join(dst_img_mode_path,fname)
    dst_path=os.path.join(dst_ann_mode_path,fname.replace('.jpg','.txt'))
    dst_file=open(dst_path,'a+')
    dst_file.write('{} {} {} {} {}\n'.format(cat_new,new_bbox[0],new_bbox[1],new_bbox[2],new_bbox[3]))

