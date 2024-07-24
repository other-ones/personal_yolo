import os
import numpy as np
import shutil
img_root='/data/twkim/doc_layout/raw/docbank/yolo/images/'
all_img_root='/data/twkim/doc_layout/raw/docbank/yolo/images/all'
flist=os.listdir(all_img_root)
modes=['train','val','test']
for mode in modes:
    mode_list_path=os.path.join('/data/twkim/doc_layout/raw/docbank/yolo/{}_c5.txt'.format(mode))
    dst_mode_list_path=os.path.join('/data/twkim/doc_layout/raw/docbank/yolo/{}_c5_v2.txt'.format(mode))
    dst_file=open(dst_mode_list_path,'w')

    lines=open(mode_list_path).readlines()
    dst_mode_root=os.path.join(img_root,mode)
    os.makedirs(dst_mode_root,exist_ok=True)
    for line in lines:
        line=line.strip()
        fname=line.split('/')[-1]
        src_path=os.path.join(all_img_root,fname)
        dst_path=os.path.join(dst_mode_root,fname)
        if not os.path.exists(src_path):
            continue
        dst_file.write('{}\n'.format(line))
        shutil.move(src_path,dst_path)