import copy
import json
import os
import shutil
import numpy as np
cls_mapping={
    "text":1,
    "table":2,
    "figure":3,
    "formula":4,
    "footer":5,
    "header":6,
}

doclaynet_mapping={
    "Caption":"text",
    "Footnote":"text",
    "Formula":"formula",#formula
    "List-item":"text",
    "Page-footer":"footer",#footer
    "Page-header":"header",
    "Picture":"figure",
    "Section-header":"text",
    "Table":"table",
    "Text":"text",
    "Title":"table",
}

root_doclaynet='/data/twkim/doc_layout/raw/doclaynet/COCO'
new_categories=[
    {"id":1,"name":"text","supercategory":''},
    {"id":2,"name":"table","supercategory":''},
    {"id":3,"name":"figure","supercategory":''},
    {"id":4,"name":"formula","supercategory":''},
    {"id":5,"name":"footer","supercategory":''},
    {"id":6,"name":"header","supercategory":''},
]



modes=['train','val','test']
for mode in modes:
    abs_count=0
    exist_count=0
    print('doclaynet loading..',mode)
    data1=json.load(open(os.path.join(root_doclaynet,'{}.json'.format(mode))))
    print('done')
    cat_list1=data1['categories']
    name_to_id_old={}
    id_to_name_old={}
    for cat in cat_list1:
        name=cat['name']
        id=cat['id']
        name_to_id_old[name]=id
        id_to_name_old[id]=name
        
    anns=data1['annotations']
    img_data=data1['images']
    ann_data=[]

    id2fname={}
    id2img={}
    print(len(anns),'len(anns)',len(img_data),'img_data')
    for img in img_data:
        id=img['id']
        # print(id,type(id),'id')
        fname=img['file_name']
        fpath=os.path.join(root_doclaynet,'../PNG',fname)
        if not os.path.exists(fpath):
            abs_count+=1
            continue
        id2fname[id]=fname
        id2img[id]=img
    header_count=0
    for idx,old_ann in enumerate(anns):
        old_id=old_ann['category_id']
        image_id=old_ann['image_id']
        if not image_id in id2fname:
            continue
        old_name=id_to_name_old[old_id]
        new_name=doclaynet_mapping[old_name]
        new_id=cls_mapping[new_name]
        if new_id==6:
            header_count+=1
        new_ann=copy.deepcopy(old_ann)
        new_ann['category_id']=new_id
        fname=id2fname[image_id]
        fpath=os.path.join(root_doclaynet,'../PNG',fname)

        if not os.path.exists(fpath):
            assert False
        else:
            exist_count+=1
            ann_data.append(new_ann)
        
    new_data={
        'categories':new_categories,
        'annotations':ann_data,
        'images':img_data,
    }
    new_file=open(os.path.join(root_doclaynet,'{}_c6.json'.format(mode)),'w')
    json.dump(new_data,new_file,indent=1)
    print('abs',abs_count)
    print('exist',exist_count)
    print(len(ann_data),'ann_data')
    print(len(img_data),'img_data')
    print('header',header_count)
    





















