import cv2
from PIL import Image
import os
import numpy as np
import shutil
import json
ann={
    "categories": [
  {
   "id": 1,
   "name": "text",
   "supercategory": ""
  },
  {
   "id": 2,
   "name": "table",
   "supercategory": ""
  },
  {
   "id": 3,
   "name": "figure",
   "supercategory": ""
  },
  {
   "id": 4,
   "name": "formula",
   "supercategory": ""
  },
  {
   "id": 5,
   "name": "footer",
   "supercategory": ""
  },
  ]
}
conf='c6'
root='/data/twkim/doc_layout/augmented1025/'
dst_root='/data/twkim/doc_layout/augmented1025'
dst_img_root=os.path.join(dst_root,'images')
dst_ann_root=os.path.join(dst_root,'labels')
os.makedirs(dst_img_root,exist_ok=True)
os.makedirs(dst_ann_root,exist_ok=True)

src_img_root=os.path.join(root,'images')
src_ann_root=os.path.join(root,'coco')
mfiles=os.listdir(src_ann_root)


val_boxes=[]
val_fname=None
for mf in ['train']:
  print('loading',mf)
  mode=mf.split('.')[0]
  dst_img_mode_path=os.path.join(dst_img_root,mode)
  dst_ann_mode_path=os.path.join(dst_ann_root,mode+'_{}'.format(conf))
  if os.path.exists(dst_ann_mode_path):
    print('delete', dst_ann_mode_path)
    shutil.rmtree(dst_ann_mode_path)
  print('done')
  os.makedirs(dst_img_mode_path,exist_ok=True)
  os.makedirs(dst_ann_mode_path,exist_ok=True)
  src_ann_file=open(os.path.join(src_ann_root,'{}_c8.json'.format(mode)))
  data=json.load(src_ann_file)
  img_data=data['images']
  
  id2fname={}
  fname2id={}
  id2img={}



  list_file=open(os.path.join(dst_root,'{}_{}.txt'.format(mode,conf)),'w')
  for img in img_data:
    img_id=img['id']
    fname=img['file_name']
    list_file.write('./images/{}/{}\n'.format(mode+'_{}'.format(conf),fname))
    src_img_path=os.path.join(src_img_root,mode+'_c8'.format(conf),fname)
    if not os.path.exists(src_img_path):
      continue
    id2fname[img_id]=fname
    fname2id[fname]=img_id
    id2img[img_id]=img


  # Parse Resolution data
  print(len(img_data),'len(img_data)')
  print(len(fname2id),'len(fname2id)')
  print(len(id2img),'len(id2img)')
  id2res={}
  res_path='res_augmented_{}.txt'.format(mode)
  if not os.path.exists(res_path):
    res_file=open('res_augmented_{}.txt'.format(mode),'w')
    res_file.write('fname\twidth\theight\n')
    for img in img_data:
      fname=img['file_name']
      src_img_path=os.path.join(src_img_root,'{}_{}'.format(mode,conf),fname)
      if not os.path.exists(src_img_path):
        continue
      img_pil=Image.open(src_img_path)
      img_width,img_height=img_pil.size
      res_file.write('{}\t{}\t{}\n'.format(fname,img_width,img_height))
      res_file.flush()
      img_id=fname2id[fname]
      id2res[img_id]=(img_width,img_height)

  else:
    res_file=open('res_augmented_{}.txt'.format(mode),'r')
    for line in res_file.readlines()[1:]:
      line=line.strip()
      splits=line.split('\t')
      fname=splits[0]
      img_width,img_height=np.array(splits[1:]).astype(np.int32)
      if not fname in fname2id:
        continue
      img_id=fname2id[fname]
      id2res[img_id]=(img_width,img_height)
  # Parse Resolution data

  
  print(len(id2res.keys()),'id2res')
  ann_data=data['annotations']
  for idx,ann in enumerate(ann_data):
    cat_id=ann['category_id']
    if cat_id in [7,8]:
      continue
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
    new_bbox=np.clip(np.array(new_bbox),0,1)
    if not image_id in id2fname:
      assert False
    fname=id2fname[image_id].split('.')[0]
    dst_path=os.path.join(dst_ann_mode_path,fname+'.txt')
    dst_file=open(dst_path,'a+')
    dst_file.write('{} {} {} {} {}\n'.format(cat_id-1,new_bbox[0],new_bbox[1],new_bbox[2],new_bbox[3]))



