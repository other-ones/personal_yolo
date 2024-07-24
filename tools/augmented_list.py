import os
import numpy as np
import shutil

confg='c6'

modes=['train','test','val']
root='/data/twkim/doc_layout/augmented1025/'
for mode in modes:
    dst_file=open(os.path.join(root,'{}_{}.txt'.format(mode,confg)),'w')
    mode_path=os.path.join(root,'images',mode+'_{}'.format(confg))
    flist=os.listdir(mode_path)
    for f in flist:
        dst_file.write('./{}\n'.format('images/{}_{}/{}'.format(mode,confg,f)))
