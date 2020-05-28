import os
import datetime
import json
import csv

def create_image_info(image_id, file_name, image_size):
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "height": image_size[0],
        "width": image_size[1],
    }

    return image_info


def create_annotation_info(annotation_id, image_id, category_id, is_crowd,
                           area, bounding_box):
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": is_crowd,
        "area": area,  # float
        "bbox": bounding_box,  # [x,y,width,height]
    }

    return annotation_info


def convert(imgdir, annpath):
    '''
    :param imgdir: directory for your images
    :param annpath: path for your annotations
    :return: coco_output is a dictionary of coco style which you could dump it into a json file
    as for keywords 'info','licenses','categories',you should modify them manually
    '''
    coco_output = {}
    coco_output['info'] = {
        "description": "Wheat Dataset",
        "url": "https://github.com/waspinator/pycococreator",
        "version": "0.1.0",
        "year": 2020,
        "contributor": "fire",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }
    coco_output['licenses'] = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]
    labels = ['wheat']

    coco_output['categories'] = []
    for i in range(len(labels)):
        coco_output['categories'].append({
            'id': i + 1,
            'name': labels[i],
        })

    coco_output['images'] = []
    coco_output['annotations'] = []
    img_id_list = {}
    img_id = -1
    ann_id = 0

    image_list = os.listdir(imgdir)
    file = open(annpath, mode='r')
    for line, row in enumerate(csv.reader(file, delimiter=',')):

        filename = row[0] + '.jpg'
        if filename not in img_id_list.keys() and filename in image_list:
            img_id += 1
            img_id_list[filename] = img_id
            image_info = create_image_info(img_id, os.path.basename(filename), [int(row[2]), int(row[1])])
            coco_output['images'].append(image_info)

        if filename in image_list:
            iscrowd = 0
            cat_id = 1
            bbox = str(row[3]).replace('[', '').replace(']', '').split(', ')
            # make annotations info and storage it in coco_output['annotations']
            x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            box = [x, y, w, h]
            area = w * h
            ann_info = create_annotation_info(ann_id, img_id, cat_id, iscrowd, area, box)
            coco_output['annotations'].append(ann_info)
            ann_id = ann_id + 1
    print(len(image_list), len(coco_output['images']), len(img_id_list), len(coco_output['annotations']))
    # exclude_img = []
    # for img in image_list:
    #     if img.split('.')[0] not in img_id_list:
    #         exclude_img.append(img)
    # print(exclude_img)
    return coco_output





coco_out = convert(imgdir='./train/', annpath='./val.csv')
with open('./instances_val.json', 'w', encoding='utf-8') as f:
    json.dump(coco_out, f, indent=4)
