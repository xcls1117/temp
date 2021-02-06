#!usr/bin/python
# -*- coding:utf-8 -*-

# 通过解析xml文件
'''
try:
    import xml.etree.CElementTree as ET
except:
    import xml.etree.ElementTree as ET

从Python3.3开始ElementTree模块会自动寻找可用的C库来加快速度
'''
import xml.etree.ElementTree as ET
import os
import sys
import cv2
from tqdm import tqdm
import shutil
import random
import numpy as np

CLASSES = ['large_luggage']  # change this label

index_map = dict(zip(CLASSES, range(len(CLASSES))))
print('index_map: ', index_map)


# label验证函数
def validate_label(xmin, ymin, xmax, ymax, width, height):
    """Validate labels."""
    # assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
    # assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
    # assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
    # assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > width:
        xmax = width
    if ymax > height:
        ymax = height

    return xmin, ymin, xmax, ymax


def load_and_save(save_path, xmls_set, xmls_path):
    for xmlFilePath in tqdm(xmls_set):
        if not xmlFilePath.endswith('.xml'):
            continue
        # print(os.path.join(xmls_path,xmlFilePath))
        try:
            tree = ET.parse(os.path.join(xmls_path, xmlFilePath))

            # 获得根节点
            root = tree.getroot()
        except Exception as e:  # 捕获除与程序退出sys.exit()相关之外的所有异常
            print("parse test.xml fail!")
            sys.exit()

        objects = root.findall("object")
        # print(objects)

        img_path = os.path.join(image_src_path, xmlFilePath.split('.')[0] + '.jpg')
        img = cv2.imread(img_path)
        if img is None:
            print('img is None: ', img_path)
        height, width = img.shape[:2]

        save_txt_path = os.path.join(save_path, xmlFilePath.split('.')[0] + '.txt')
        f = open(save_txt_path, 'w')
        # print(f)
        no_valid_class = True
        string = ''

        for obj in objects:
            cls_name = obj.find('name')
            if cls_name is None:
                print('cls_name is None')
                continue
            cls_name = cls_name.text.lower().strip()
            if cls_name not in CLASSES:
                print('new cla_name: ', cls_name)
                continue

            cls_id = index_map[cls_name]

            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text))
            ymin = (float(xml_box.find('ymin').text))
            xmax = (float(xml_box.find('xmax').text))
            ymax = (float(xml_box.find('ymax').text))

            try:
                xmin, ymin, xmax, ymax = validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                print('class {}, xmin {}, ymin {}, xmax {}, ymax {}, width {}, height {}'.format(cls_name, xmin, ymin,
                                                                                                 xmax, ymax, width,
                                                                                                 height))
                print('error image: ', img_path)
                # raise RuntimeError("Invalid label at {}, {}".format(xmlFilePath, e))

            w = xmax - xmin
            h = ymax - ymin
            # if w<12 or h<12:
            #     continue

            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            w_norm = w * 1.0 / width
            h_norm = h * 1.0 / height

            no_valid_class = False

            string += str(cls_id) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w_norm) + ' ' + str(
                h_norm) + '\n'

        print(string)
        if no_valid_class:
            continue
        else:
            f.write(string)

    f.close()


def copy_image(image_ori_path, image_src_path, xmls_path):
    if not os.path.exists(image_src_path):
        os.makedirs(image_src_path)
    if not os.path.exists(xmls_path):
        os.makedirs(xmls_path)
    for file_path in os.listdir(image_ori_path):
        for filename in os.listdir(os.path.join(image_ori_path, file_path)):
            if filename.endswith('.xml'):
                shutil.copy(os.path.join(image_ori_path, file_path, filename), xmls_path)
            else:
                shutil.copy(os.path.join(image_ori_path, file_path, filename), image_src_path)


def split_train_val(image_src_path, txt_path, target_path):
    dirname = target_path
    train_img_path = os.path.join(dirname,'images', 'train')
    train_label_path = os.path.join(dirname,'labels','train')
    val_img_path = os.path.join(dirname, 'images', 'val')
    val_label_path = os.path.join(dirname, 'labels', 'val')
    if not os.path.exists(train_img_path):
        os.makedirs(train_img_path)
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)
    if not os.path.exists(val_img_path):
        os.makedirs(val_img_path)
    if not os.path.exists(val_label_path):
        os.makedirs(val_label_path)

    filelists = os.listdir(image_src_path)
    random.seed(42)
    random.shuffle(filelists)
    train_ratio = 0.9
    train_set = filelists[:int(len(filelists)*train_ratio)]
    val_set = filelists[int(len(filelists)*train_ratio):]
    for filename in train_set:
        shutil.move(os.path.join(image_src_path, filename), train_img_path)
        txtname = filename.replace('.jpg', '.txt')
        shutil.move(os.path.join(txt_path, txtname), train_label_path)
    for filename in val_set:
        shutil.move(os.path.join(image_src_path, filename), val_img_path)
        txtname = filename.replace('.jpg', '.txt')
        shutil.move(os.path.join(txt_path, txtname), val_label_path)


if __name__ == "__main__":
    image_ori_path = sys.argv[1]
    image_src_path = sys.argv[2]  # image path
    xmls_path = sys.argv[3]  # xml path
    target_txt_path = sys.argv[4]  # output
    target_train_path = sys.argv[5]  # output

    copy_image(image_ori_path, image_src_path, xmls_path)
    if not os.path.exists(target_txt_path):
        os.makedirs(target_txt_path)
    if not os.path.exists(target_train_path):
        os.makedirs(target_train_path)

    allxmls = os.listdir(xmls_path)
    
    load_and_save(target_txt_path, allxmls, xmls_path)
    split_train_val(image_src_path, target_txt_path, target_train_path)
