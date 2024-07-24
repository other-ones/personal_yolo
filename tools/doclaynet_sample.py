import os
import numpy as np
import shutil


mode='test'
root='/data/twkim/doc_layout/raw/doclaynet/yolo/'
val_list_path=os.path.join(root,'{}_c6_sampled.txt'.format(mode))
dst_file=open(dst_path,'w')
val_lines=open(val_list_path).readlines()
np.random.shuffle(val_lines)
for line in val_lines[:1000]:
    line=line.strip()
    dst_file.write('{}\n'.format(line))
    
