#!/bin/bash

log_file=/project/train/log/log.txt
dataset_dir=../preprocessed_dataset

if [ ! -d "/project/train/log/" ]; then
  mkdir /project/train/log/
fi

cd /project/train/src_repo

echo "Preparing..."
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/

# process data
echo "Converting dataset..."
python3 preprocess.py /home/data ${dataset_dir}/images ${dataset_dir}/xmls ${dataset_dir}/txts ${dataset_dir}/train

# start train
echo "Start training..."
python3 train.py --data ./data/cus.yaml --weights yolov5s.pt --workers 2 --epochs 100 --img 960 2>&1 | tee -a ${log_file} 
