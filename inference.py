import os
import cv2
import re
import pandas as pd
from modules import Preprocess, Detection, OCR, Correction
from tool.utils import natural_keys, visualize
import time
import matplotlib.pyplot as plt


img_path = 'our_data/cm_2.JPG'
result_path = 'result'
if not os.path.exists(result_path):
    os.makedirs(result_path)

det_weight = 'weights/PANNet_best_map.pth'
ocr_weight = 'weights/transformerocr.pth'

img = cv2.imread(img_path)

cv2.imwrite(os.path.join(result_path, "anh_goc.png"), img)

det_model = Detection(weight_path=det_weight)
ocr_model = OCR(weight_path=ocr_weight)
preproc = Preprocess(
    det_model=det_model,
    ocr_model=ocr_model,
    find_best_rotation=True)
correction = Correction()

img1 = preproc(img)

cv2.imwrite(os.path.join(result_path, "anh_cat_goc.png"), img1)

boxes, img2  = det_model(
    img1,
    crop_region=True,                               #Crop detected regions for OCR
    return_result=True,                             # Return plotted result
    output_path=result_path   #Path to save cropped regions
)
cv2.imwrite(os.path.join(result_path, "anh_co_cac_o.png"), img2)


img_paths=os.listdir(os.path.join(result_path,'crops')) # Cropped regions
img_paths.sort(key=natural_keys)
img_paths = [os.path.join(os.path.join(result_path,'crops'), i) for i in img_paths]

texts, probs = ocr_model.predict_folder(img_paths, return_probs=True) # OCR
texts = correction(texts)   # Word correction

count = 0

with open(os.path.join(result_path,'ocr.txt'),"w") as f:
    for i in texts:
        f.write(str(count) + " - " + i + "\n")
        count +=1