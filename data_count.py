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
import numpy as np

#label验证函数
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

def get_all_class(xmls_set,xmls_path):
    ret = []
    for xmlFilePath in tqdm(xmls_set):
        if not xmlFilePath.endswith('.xml'):
            continue
        # print(os.path.join(xmls_path,xmlFilePath))
        try:
            tree = ET.parse(os.path.join(xmls_path,xmlFilePath))

            # 获得根节点
            root = tree.getroot()
        except Exception as e:  # 捕获除与程序退出sys.exit()相关之外的所有异常
            print("parse test.xml fail!")
            sys.exit()

        objects = root.findall("object")

        for obj in objects:
            cls_name = obj.find('name')
            if cls_name is None:
                print('cls_name is None')
                continue
            cls_name = cls_name.text.lower().strip()
            if cls_name not in ret:
                print('new cla_name: ', cls_name)
                
                ret.append(cls_name)
    
    print('CLASSES: ', ret)
    return ret


def count(xmls_set,xmls_path):
    CLASSES = get_all_class(xmls_set,xmls_path)
    ret = np.zeros(len(CLASSES))

    for xmlFilePath in tqdm(xmls_set):
        if not xmlFilePath.endswith('.xml'):
            continue
        # print(os.path.join(xmls_path,xmlFilePath))
        try:
            tree = ET.parse(os.path.join(xmls_path,xmlFilePath))

            # 获得根节点
            root = tree.getroot()
        except Exception as e:  # 捕获除与程序退出sys.exit()相关之外的所有异常
            print("parse test.xml fail!")
            sys.exit()

        objects = root.findall("object")

        for obj in objects:
            cls_name = obj.find('name')
            if cls_name is None:
                print('cls_name is None')
                continue
            cls_name = cls_name.text.lower().strip()
            if cls_name not in CLASSES:
                print('new cla_name: ', cls_name)
                continue
            
            ret[CLASSES.index(cls_name)] += 1
        
    print('count ret: ', ret)
    for i in range(len(CLASSES)):
        print('%s\t%d'%(CLASSES[i], ret[i]))

if __name__ == "__main__":
    image_src_path = sys.argv[1]
    xmls_path = sys.argv[2]

    allxmls = os.listdir(xmls_path)

    count(allxmls, xmls_path)

