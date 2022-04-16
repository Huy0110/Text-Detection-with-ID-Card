# Text detection with ID Card
This repo references to https://github.com/kaylode/vietnamese-ocr-toolbox .<br />

I has fixed and add some file to make the model fit with new data of Vietname ID Card and clearing the code

# Requirement
```bash
pip install -r requirements.txt
```

# Dataset
New dataset of VietNam ID card with 7 class, corresponds to 7 folders.<br />

In each forders there are images accompanying the corresponding json file (with the same name)
The example dataset:

# Run

## Process with new data:

### 1. Process data to train, val, test forders
```bash
python data_process.py
```

### 2. Prepare data to get txt files
```bash
python data_prepare_2.py
```

### 3. Convert data to the formated data
```bash
python convert.py
```

### 4. Move data_formated forder to data forder
```bash
mv data_formated data
```

## Download pre-trained weights
```bash
cd weights
```
You should already have gdrive
```bash
gdrive download 1GKs-NnezTc6WN0P_Zw7h6wYzRRZdJFKl
```
## Train
```bash
python train.py
```

## Inference
```bash
python inference.py --weight 'weights/PANNet_best_map.pth'
```



