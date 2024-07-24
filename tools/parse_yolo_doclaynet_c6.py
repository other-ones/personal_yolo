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
  }]
}
conf='c6'
root='/data/twkim/doc_layout/raw/doclaynet'
dst_root='/data/twkim/doc_layout/raw/doclaynet/yolo'
dst_img_root=os.path.join(dst_root,'images')
dst_ann_root=os.path.join(dst_root,'labels')
os.makedirs(dst_img_root,exist_ok=True)
os.makedirs(dst_ann_root,exist_ok=True)

src_img_root=os.path.join(root,'PNG')
src_ann_root=os.path.join(root,'COCO')
mfiles=os.listdir(src_ann_root)



val_boxes=[]
val_fname=None
for mf in ['train','val','test']:

  # if not 'val' in mf:
  #   continue
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
  src_ann_file=open(os.path.join(src_ann_root,'{}_{}.json'.format(mode,conf)))
  data=json.load(src_ann_file)
  img_data=data['images']
  id2fname={}
  fname2id={}
  id2img={}



  list_file=open(os.path.join(dst_root,'{}_c6.txt'.format(mode)),'w')
  for img in img_data:
    img_id=img['id']
    fname=img['file_name']
    list_file.write('./images/{}/{}\n'.format(mode+'_c6',fname))
    fpath=os.path.join(src_img_root,fname)
    if not os.path.exists(fpath):
      continue
    id2fname[img_id]=fname
    fname2id[fname]=img_id
    id2img[img_id]=img


  # Parse Resolution data
  id2res={}
  res_path='res_doclaynet_{}.txt'.format(mode)
  if not os.path.exists(res_path):
    res_file=open('res_doclaynet_{}.txt'.format(mode),'w')
    res_file.write('fname\twidth\theight\n')
    for img in img_data:
      fname=img['file_name']
      fpath=os.path.join(src_img_root,fname)
      img_pil=Image.open(fpath)
      img_width,img_height=img_pil.size
      res_file.write('{}\t{}\t{}\n'.format(fname,img_width,img_height))
      img_id=fname2id[fname]
      id2res[img_id]=(img_width,img_height)

  else:
    res_file=open('res_doclaynet_{}.txt'.format(mode),'r')
    for line in res_file.readlines()[1:]:
      line=line.strip()
      splits=line.split('\t')
      fname=splits[0]
      img_width,img_height=np.array(splits[1:]).astype(np.int32)
      img_id=fname2id[fname]
      id2res[img_id]=(img_width,img_height)
  # Parse Resolution data


  
  ann_data=data['annotations']
  for idx,ann in enumerate(ann_data):
    cat_id=ann['category_id']
    cat=cat_id-1
    if cat in [0,3]:
      cat=0
    elif cat==1:
      cat=1
    elif cat==2:
      cat=2
    elif cat==4:
      cat=4
    image_id=ann['image_id']
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
    if not image_id in id2fname:
      assert False
    fname=id2fname[image_id].split('.')[0]
    src_image_path=os.path.join(src_img_root,fname+'.png')
    dst_image_path=os.path.join(dst_img_mode_path,fname+'.png')
    if not os.path.exists(dst_image_path):
      os.symlink(src_image_path,dst_image_path)
    dst_path=os.path.join(dst_ann_mode_path,fname+'.txt')
    dst_file=open(dst_path,'a+')
    dst_file.write('{} {} {} {} {}\n'.format(cat_id-1,new_bbox[0],new_bbox[1],new_bbox[2],new_bbox[3]))

    # if 'aa6ab783f94cdf57b4a82c6dc28ee08e3de39cc1cfae542ca69834cfa90233cb' in fname:
    #   val_boxes.append(np.array(new_bbox))
    #   val_fname=fname


# def visualize_box(image,boxes,chars=None):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     fontScale = 0.6
#     image=np.array(image)
#     if chars is not None:
#         for box,char in zip(boxes,chars):
#             box=(box.detach().cpu().numpy()*2).astype(np.int32)
#             x0,y0,x1,y1=box
#             box=(box).reshape(-1,2).astype(np.int32)
#             point1=box[0]
#             point2=box[1]
#             image=cv2.rectangle(image,tuple(point1),tuple(point2),color=(0,255,0),thickness=1)   
#             image = cv2.putText(image, char, (x0,y1+5), font, fontScale, (255,0,0), 1, cv2.LINE_AA)
#     else:
#         for box in boxes:
#             box=np.array(box).reshape(-1,2)
#             point1=box[0]
#             point2=box[1]
#             image=cv2.rectangle(image,tuple(point1),tuple(point2),color=(0,255,0),thickness=1)
#     image=Image.fromarray(image)
#     return image






# img_pil=Image.open(os.path.join(src_img_root,val_fname+'.png'))
# new_boxes=[]
# for box in val_boxes:
#   center_x,center_y,box_w,box_h=box
#   x0=center_x-box_w/2
#   y0=center_y-box_h/2
#   x1=center_x+box_w/2
#   y1=center_y+box_h/2
#   new_boxes.append(np.array([x0*1025,y0*1025,x1*1025,y1*1025]).astype(np.int32))
# drawn_val=visualize_box(img_pil,new_boxes)
# drawn_val.save('drawn_val.jpg')

    



