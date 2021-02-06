#!/bin/bash

mkdir /project/train/log

cd  /project/train/src_repo/

pip3 install -r requirements.txt
# process data
python3 preprocess.py /home/data ../preprocessed_dataset/images ../preprocessed_dataset/xmls ../preprocessed_dataset/txts ../preprocessed_dataset/train

# start train
python3 train.py --data ./data/cus.yaml --weights yolov5s.pt --workers 2 --epochs 100 --img 960 2>&1 | tee /project/train/log/log.txt
