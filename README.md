# Text detection with ID Card
This repo references to https://github.com/kaylode/vietnamese-ocr-toolbox .<br />

I has fixed and add some file to make the model fit with new data of Vietname ID Card and clearing the code

# Requirement
```bash
pip install -r requirements.txt
```

# Dataset
New dataset of VietNam ID card with 7 class, corresponds to 7 folders.<br />

In each forders there are images accompanying the corresponding json file (with the same name).<br />
The example dataset: https://drive.google.com/drive/folders/12S7zZPQc0rkKgSEsxCnAlMpaI2oqKFGH?usp=sharing

# Run

## 1. Process with new data:

### Process data to train, val, test forders
```bash
python data_process.py
```

### Prepare data to get txt files
```bash
python data_prepare_2.py
```

### Convert data to the formated data
```bash
python convert.py
```

### Move data_formated forder to data forder
```bash
mv data_formated data
```

## 2. Download pre-trained weights
```bash
cd weights
```
You should already have gdrive
```bash
gdrive download 1GKs-NnezTc6WN0P_Zw7h6wYzRRZdJFKl
```
## 3. Train
```bash
python train.py
```

## 4. Inference
```bash
python inference.py --weight 'weights/PANNet_best_map.pth'
```



