import os
import numpy as np
lines=open('/data/twkim/doc_layout/raw/docbank/yolo/val_c5_v2.txt').readlines()
np.random.shuffle(lines)

sampled=lines[:5000]
dst_file=open('/data/twkim/doc_layout/raw/docbank/yolo/val_sampled_c5_v2.txt','w')
for line in sorted(sampled):
    line=line.strip()
    dst_file.write('{}\n'.format(line))