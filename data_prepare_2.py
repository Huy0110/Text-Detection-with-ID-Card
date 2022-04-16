import json
import os
from tqdm import tqdm

list_data_dir = ['train', 'val', 'test']
for dir in list_data_dir:
    data_dir = os.path.join(dir, 'json')
    txt_dir = os.path.join(dir, 'txt')
    paths = os.listdir(data_dir)
    paths = [i for i in paths if i.endswith(".json")]
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    for image_id, image_ann_path in tqdm(enumerate(paths)):
        img_name = image_ann_path[:-4] + "png"
        ann_path = os.path.join(data_dir, image_ann_path)
        img_path = os.path.join(data_dir, img_name)
        txt_path = os.path.join(txt_dir, image_ann_path[:-4] + "txt")
        with open(ann_path, "r") as js:
            data = json.load(js)
        for elem in data:
            pre_length = 0
            txt = elem['text']
            td1, td2, td3, td4 = elem['polygon']
            xx1, yy1 = td1
            xx2, yy2 = td2
            xx3, yy3 = td3
            xx4, yy4 = td4
            with open(txt_path, 'a', encoding='utf-8') as f:
                f.write(str(xx1) + ',' + str(yy1) + ',' + str(xx2) + ',' + str(yy2) + ',' +str(xx3) + ',' + str(yy3) + ',' + str(xx4) + ',' + str(yy4) + ',' + txt + '\n')
