# coding=utf-8
# 从XML 读取ground truth坐标

import xml.dom.minidom

#class_map = {0: 'label0', 1: 'label1', 2: 'label2'}
class_map={"aircraft":'1'}  # 只有一类的飞机目标
def parse_xml(xml_path):
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    objects = root.getElementsByTagName('object')
    gts = []
    for index, obj in enumerate(objects):
        name = obj.getElementsByTagName('name')[0].firstChild.data
        label = class_map[name]
        bndbox = obj.getElementsByTagName('bndbox')[0]
        x1 = int(bndbox.getElementsByTagName('xmin')[0].firstChild.data)
        y1 = int(bndbox.getElementsByTagName('ymin')[0].firstChild.data)
        x2 = int(bndbox.getElementsByTagName('xmax')[0].firstChild.data)
        y2 = int(bndbox.getElementsByTagName('ymax')[0].firstChild.data)
        gt_one = [label, x1, y1, x2, y2]
        gts.append(gt_one)
    return gts