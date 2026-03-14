# coding=utf-8
import multiprocessing
import os
import xml.etree.ElementTree
from pathlib import Path

label_dir = "D:\\Safety Helmet Wearing Dataset\\Safety Helmet Wearing Dataset\\Annotations"
save_dir = "D:\\Safety Helmet Wearing Dataset\\Safety Helmet Wearing Dataset\\Images"
names = ["helmet", "head_with_helmet", "person_with_helmet", "head", "person_no_helmet", "face"]
names = ["head", "head_with_helmet", "person_with_helmet", "person_no_helmet",]

def voc2yolo(xml_file):
    in_file = open(f'{label_dir}/{xml_file}', encoding='utf-8')

    root = xml.etree.ElementTree.parse(in_file).getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    has_class = False
    for obj in root.iter('object'):
        name = obj.find('name').text
        if name in names:
            has_class = True
    if has_class:
        Path(save_dir).mkdir(exist_ok=True)
        out_file = open(f'{save_dir}/{xml_file[:-4]}.txt', 'w')
        for obj in root.iter('object'):
            name = obj.find('name').text
            if name in names:
                xml_box = obj.find('bndbox')
                x_min = float(xml_box.find('xmin').text)
                y_min = float(xml_box.find('ymin').text)
                # print(in_file)
                x_max = float(xml_box.find('xmax').text)
                y_max = float(xml_box.find('ymax').text)

                box_x = (x_min + x_max) / 2.0 - 1
                box_y = (y_min + y_max) / 2.0 - 1
                box_w = x_max - x_min
                box_h = y_max - y_min

                box_x = box_x * 1. / w
                box_w = box_w * 1. / w
                box_y = box_y * 1. / h
                box_h = box_h * 1. / h

                b = [box_x, box_y, box_w, box_h]
                cls_id = names.index(obj.find('name').text)
                out_file.write(str(cls_id) + " " + " ".join([str(f'{a:.6f}') for a in b]) + '\n')


if __name__ == '__main__':

    print('VOC to YOLO')
    xml_files = [name for name in os.listdir(label_dir) if name.endswith('.xml')]

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.map(voc2yolo, xml_files)
    pool.join()