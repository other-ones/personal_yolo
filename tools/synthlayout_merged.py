import os
import numpy as np
import shutil

confg='c5'

modes=['train','val','test']
root='/data/twkim/doc_layout/synthlayout_merged/'
img_root=os.path.join(root,'images','raw')
flist=os.listdir(img_root)
np.random.shuffle(flist)
val_list=flist[:1000]
test_list=flist[1000:2000]
train_list=flist[2000:]
flists=[val_list,test_list,train_list]

dst_file=open(os.path.join(root,'val_{}.txt'.format(confg)),'w')
for f in val_list:
    dst_file.write('./{}\n'.format('images/raw/{}'.format(f)))

dst_file=open(os.path.join(root,'test_{}.txt'.format(confg)),'w')
for f in test_list:
    dst_file.write('./{}\n'.format('images/raw/{}'.format(f)))


dst_file=open(os.path.join(root,'train_{}.txt'.format(confg)),'w')
for f in train_list:
    dst_file.write('./{}\n'.format('images/raw/{}'.format(f)))

