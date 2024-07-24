import os
import numpy as np
import shutil
import json



src_root1='/data/twkim/doc_layout/synthlayout_english'
src_root2='/data/twkim/doc_layout/synthlayout_mlt'

img_root1=os.path.join(src_root1,'images/train_c6')
lab_root1=os.path.join(src_root1,'labels/train_c6')
flist1=os.listdir(img_root1)
fnames1=[]
for item in flist1:
    fname=item.split('.')[0]
    label_path=os.path.join(lab_root1,fname+'.txt')
    if os.path.exists(label_path):
        fnames1.append(fname)

img_root2=os.path.join(src_root2,'images/train_c6')
lab_root2=os.path.join(src_root2,'labels/train_c6')
flist2=os.listdir(img_root2)
fnames2=[]
for item in flist2:
    fname=item.split('.')[0]
    label_path=os.path.join(lab_root2,fname+'.txt')
    if os.path.exists(label_path):
        fnames2.append(fname)

dst_root='/data/twkim/doc_layout/synthlayout_merged'
dst_img_root=os.path.join(dst_root,'images/train')
dst_lab_root=os.path.join(dst_root,'labels/train')
os.makedirs(dst_img_root,exist_ok=True)
os.makedirs(dst_lab_root,exist_ok=True)
# 1. list items
fnames_list=[fnames1,fnames2]
roots=[src_root1,src_root2]
count=0
for fnames,root in zip(fnames_list,roots):
    print(root)
    for fname in fnames:
        src_img_path=os.path.join(root,'images/train_c6',fname+'.png')
        src_lab_path=os.path.join(root,'images/train_c6',fname+'.txt')
        dst_img_path=os.path.join(dst_img_root,'images/train_c6','{:06d}.jpg'.format(count+1))
        dst_lab_path=os.path.join(dst_img_root,'images/train_c6','{:06d}.txt'.format(count+1))
        src_lab_lines=open(src_lab_path).readlines()
        for line in src_lab_lines:
            line=line.strip()
            splits=line.split()
            # correct labels
            # text:0 / figure:1 / table:2
        count+=1


# 2. merge images
# 3. mergge labels 
# correct labels