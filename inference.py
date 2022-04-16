import os
import cv2
import re
import pandas as pd
from modules import Preprocess, Detection, OCR, Correction
from tool.utils import natural_keys, visualize
import time
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weight', required=True, type=str)

det_weight = vars(parser.parse_args())['weight']


img_path = 'our_data/cm_2.JPG'
result_path = 'result'
if not os.path.exists(result_path):
    os.makedirs(result_path)

#det_weight = 'weights/PANNet_best_map.pth' #your weight path

img = cv2.imread(img_path)

cv2.imwrite(os.path.join(result_path, "anh_goc.png"), img)

det_model = Detection(weight_path=det_weight)

img1 = img

boxes, img2  = det_model(
    img1,
    crop_region=True,                               #Crop detected regions for OCR
    return_result=True,                             # Return plotted result
    output_path=result_path   #Path to save cropped regions
)
cv2.imwrite(os.path.join(result_path, "final_detect.png"), img2)

img_paths=os.listdir(os.path.join(result_path,'crops')) # Cropped regions
img_paths.sort(key=natural_keys)
img_paths = [os.path.join(os.path.join(result_path,'crops'), i) for i in img_paths]