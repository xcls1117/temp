# /usr/bin/python
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

CLASSES = ['large_luggage']  # change this label

index_map = dict(zip(CLASSES,range(len(CLASSES))))
print('index_map: ', index_map)

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

def load_and_save(save_path,xmls_set,xmls_path):
    
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
        
        img_path = os.path.join(image_src_path, xmlFilePath.split('.')[0] + '.jpg')
        img = cv2.imread(img_path)
        if img is None:
            print('img is None: ', img_path)
        height, width = img.shape[:2]

        save_txt_path = os.path.join(save_path, xmlFilePath.split('.')[0]+'.txt')
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
                xmin,ymin,xmax,ymax=validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                print('class {}, xmin {}, ymin {}, xmax {}, ymax {}, width {}, height {}'.format(cls_name, xmin, ymin, xmax, ymax, width, height))
                print('error image: ', img_path)
                # raise RuntimeError("Invalid label at {}, {}".format(xmlFilePath, e))
            
            w = xmax - xmin
            h = ymax - ymin
            # if w<12 or h<12:
            #     continue
            
            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            w_norm = w *1.0 / width
            h_norm = h *1.0 / height

            no_valid_class = False
            
            string += str(cls_id) + ' ' + str(x_center)+' '+str(y_center)+' '+str(w_norm)+' '+str(h_norm) + '\n'
        
        print(string)
        if no_valid_class:
            continue
        else:
            f.write(string)
            
    f.close()

if __name__ == "__main__":
    image_src_path = sys.argv[1]  # image path
    xmls_path = sys.argv[2]       # xml path
    target_path = sys.argv[3]     # output

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    allxmls = os.listdir(xmls_path)

    load_and_save(target_path, allxmls, xmls_path)



