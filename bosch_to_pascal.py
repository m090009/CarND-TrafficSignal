#!/usr/bin/env python
"""
This script Converts Yaml annotations to Pascal .xml Files
of the Bosch Small Traffic Lights Dataset.
Example usage:
    python bosch_to_pascal.py input_yaml out_folder
"""

import os
import sys
import yaml
from lxml import etree
import os.path
import xml.etree.cElementTree as ET
import json
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str


def write_xml(savedir, image, imgWidth, imgHeight,
              depth=3, pose="Unspecified"):

    boxes = image['boxes']
    impath = image['path']
    imagename = impath.split('/')[-1]
    currentfolder = savedir.split("\\")[-1]
    annotation = ET.Element("annotaion")
    ET.SubElement(annotation, 'folder').text = str(currentfolder)
    ET.SubElement(annotation, 'filename').text = str(imagename)
    imagename = imagename.split('.')[0]
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(imgWidth)
    ET.SubElement(size, 'height').text = str(imgHeight)
    ET.SubElement(size, 'depth').text = str(depth)
    ET.SubElement(annotation, 'segmented').text = '0'
    for box in boxes:
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = str(box['label'])
        ET.SubElement(obj, 'pose').text = str(pose)
        ET.SubElement(obj, 'occluded').text = str(box['occluded'])
        ET.SubElement(obj, 'difficult').text = '0'

        bbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(box['x_min'])
        ET.SubElement(bbox, 'ymin').text = str(box['y_min'])
        ET.SubElement(bbox, 'xmax').text = str(box['x_max'])
        ET.SubElement(bbox, 'ymax').text = str(box['y_max'])

    xml_str = ET.tostring(annotation)
    root = etree.fromstring(xml_str)
    xml_str = etree.tostring(root, pretty_print=True)
    save_path = os.path.join(savedir, imagename + ".xml")
    with open(save_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)

def write_json(savedir, image, img_width, img_height, depth=3, pose="unspecified"):
    boxes = image['boxes']
    impath = image['path']
    imagename = impath.split('/')[-1]
    currentfolder = savedir.split("\\")[-1]
    
    objs = []
    # Getting boxes 
    for box in boxes:
            
        bndbox = {'xmax': str(box['x_max']),
                  'xmin': str(box['x_min']),
                  'ymax': str(box['y_max']),
                  'ymin': str(box['y_min'])}
    
        obj = {'name': str(box['label']),
               'pose': str(pose),
               'occluded': str(box['occluded']) ,
               'difficult': '0',
               'bndbox': bndbox}
    
        objs.append(obj)
    
    
    size = {'width': str(img_width), 
            'height': str(img_height), 
            'depth': str(depth)}
    
    img_dict = {'folder': str(currentfolder + impath.replace('./', '')),
                'filename': str(imagename),
                'size': size,
                'segmented': '0', 
                'objects': objs}
    return img_dict
        
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(-1)
    yaml_path = sys.argv[1]
    out_dir = sys.argv[2]
    dataset_type = sys.argv[3]
    images = yaml.load(open(yaml_path, 'rb').read())
#     print(write_json(out_dir, images[2], 1280, 720, depth=3, pose="Unspecified"))
    
    annotations = []

    for image in images:
        annotations.append(write_json(out_dir, image, 1280, 720, depth=3, pose="Unspecified"))
#     print(annotations[:3])
# Write JSON file
    with io.open( f'{out_dir}{dataset_type}_annotations.json', 'w', encoding='utf8') as outfile:
        str_ = json.dumps(annotations,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))