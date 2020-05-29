import os
import cv2
import random
import numpy as np # linear algebra
import pandas as pd
import glob
#from utils import postprocess_boxes
from train import efficientdet
from matplotlib import pyplot as plt
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
#from ensemble_boxes import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def draw_boxes(image, boxes, scores, labels, colors, classes):
    for b, l, s in zip(boxes, labels, scores):
        class_id = int(l)
        class_name = classes[class_id]
    
        xmin, ymin, xmax, ymax = list(map(int, b))
        score = '{:.4f}'.format(s)
        color = colors[class_id]
        label = '-'.join([class_name, score])
    
        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)











def py_nms(dets, scores, thresh, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    box_count = 0
    for i in range(1,len(dets)):
        if sum(dets[i])==0:
            box_count = i
            break

    x1 = dets[:box_count, 0]
    y1 = dets[:box_count, 1]
    x2 = dets[:box_count, 2]
    y2 = dets[:box_count, 3]
    scores = scores[:box_count]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        #keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def preprocess_image(image, image_size):
    # image, RGB
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size

    image = cv2.resize(image, (resized_width, resized_height))
    image = image.astype(np.float32)
    image /= 255.
    mean = [0.485, 0.456, 0.406] 
    # [80.29009934708377, 80.78969818174566, 54.61127266854104]
    # [0.315, 0.317, 0.214]
    std = [0.229, 0.224, 0.225] 
    # [62.060403994017065, 60.21645107360893, 48.89899595854723]
    # [0.243, 0.236, 0.192]
    image -= mean
    image /= std
    pad_h = image_size - resized_height
    pad_w = image_size - resized_width
    image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')

    return image, scale

def postprocess_boxes(boxes, scale, height, width):
    boxes /= scale
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)
    return boxes


def readData(data_path = "../data/wheat"):
    df = pd.read_csv(os.path.join(data_path, "val.csv"))
    img_names = list(set(df['image_id'].values))
    print(len(img_names))#337

    def _strToList(st):
        return [int(float(x)) for x in st[1:-1].split(',')]

    data_dict = {}
    for img_name in img_names:
        df_imgs = df[df['image_id']==img_name]
        boxes = df_imgs['bbox'].values
        data_dict[img_name] = [_strToList(x) for x in boxes]
        
    return data_dict


def bb_intersection_over_union(A, B):
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()
    for t in range(len(boxes)):
        for j in range(len(boxes[t])):
            label = int(labels[t][j])
            score = scores[t][j]
            if score < thr:
                break
            box_part = boxes[t][j]
            b = [int(label), float(score) * weights[t], float(box_part[0]), float(box_part[1]), float(box_part[2]), float(box_part[3])]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse 
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box
    """

    box = np.zeros(6, dtype=np.float32)
    conf = 0
    conf_list = []
    for b in boxes:
        box[2:] += (b[1] * b[2:])
        conf += b[1]
        conf_list.append(b[1])
    box[0] = boxes[0][0]
    if conf_type == 'avg':
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2:] /= conf
    return box


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union(box[2:], new_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0, conf_type='avg', allows_overflow=False):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers. 
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model 
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param intersection_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable  
    :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value
    :param allows_overflow: false if we want confidence score not exceed 1.0 
    
    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2). 
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max']:
        print('Unknown conf_type: {}. Must be "avg" or "max"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes.append(boxes[j].copy())

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            if not allows_overflow:
                weighted_boxes[i][1] = weighted_boxes[i][1] * min(weights.sum(), len(new_boxes[i])) / weights.sum()
            else:
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(new_boxes[i]) / weights.sum()
        overall_boxes.append(np.array(weighted_boxes))

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 2:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels


def IOU(Reframe,GTframe):
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]
    height1 = Reframe[3]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]
    height2 = GTframe[3]

    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1+width2-(endx-startx)

    endy = max(y1+height1,y2+height2)
    starty = min(y1,y2)
    height = height1+height2-(endy-starty)

    if width <=0 or height <= 0:
        ratio = 0 # 重叠率为 0 
    else:
        Area = width*height # 两矩形相交面积
        Area1 = width1*height1
        Area2 = width2*height2
        ratio = Area*1./(Area1+Area2-Area)
    # return IOU
    return ratio

def computeScore(img_boxes, pre_boxes):
    threshold_values = [x/100.0 for x in range(50,76,5)]
    scores = []

    for threshold_value in threshold_values:
        total_true = len(img_boxes)
        TP = 0
        FP = 0
        FN = 0
        find_flag = False
        pre_boxes_copy = pre_boxes.copy()
        for img_box in img_boxes:
            for i,pre_box in enumerate(pre_boxes_copy):
                iou_score = IOU(img_box, [int(x) for x in pre_box[1:]])
                if iou_score>threshold_value:
                    find_flag = True
                    if i == 0:
                        pre_boxes_copy = pre_boxes_copy[1:]
                    elif i==len(img_boxes)-1:
                        pre_boxes_copy = pre_boxes_copy[:-1]
                    else:
                        #print(pre_boxes[:i],pre_boxes[i+1:])
                        pre_boxes_copy = pre_boxes_copy[:i]+pre_boxes_copy[i+1:]
                    break
            if find_flag:
                TP+=1
            else:
                FN+=1
            find_flag = False

        FN = len(pre_boxes_copy)
        score = TP/(TP+FP+FN)
        # print(TP, FP, FN)
        #print(score)
        # b
        scores.append(score)
    #print(np.mean(scores))
    #b
    return np.mean(scores)


with open(r'../data/wheat/instances_val.json','r',encoding='utf8')as fp:
    name_json = json.load(fp)

raw_data = name_json['images']
name_dict = {}
for data_map in raw_data:
    #print(data_map['id'],data_map['file_name'])
    name_dict[data_map['file_name'].split(".")[0]] = data_map['id']



def evaluate(data_dict, data_path):
    print("start...")
    phi = 3
    weighted_bifpn = False
    
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = 1024#image_sizes[phi]
    classes = {0:"wheat"}
    num_classes = len(classes)
    score_threshold = 0.6
    colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    _, model = efficientdet(phi=phi,
                            num_classes=num_classes,
                            score_threshold=score_threshold,
                            weighted_bifpn=weighted_bifpn,
                           freeze_bn=False,
                           detect_quadrangle=False)

    model_path = "models/d3_1024.h5"
    model.load_weights(model_path, by_name=True)
    print("finish load model")
    
    results = []
    results_dict = []
    for img_name, img_boxes in data_dict.items():
        image = cv2.imread(os.path.join(data_path, img_name+".jpg"))
        # BGR -> RGB
        image = image[:, :, ::-1]
        src_image = image.copy()
        h, w = image.shape[:2]
        image, scale = preprocess_image(image, image_size=image_size)
        boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes_tmp, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        boxes = boxes_tmp.copy()
        boxes= postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)


        # nms
        boxes_keep_index = py_nms(boxes, scores, 0.5)
        boxes = boxes[boxes_keep_index]
        scores = scores[boxes_keep_index]
        labels = labels[boxes_keep_index]

        # wbf
        #boxes, scores, labels = nms([boxes], [scores], [labels], iou_thr=0.5)
        #boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels], iou_thr=0.5)
        #print(boxes.shape, scores.shape, labels.shape)

        indices = np.where(scores[:] > score_threshold)[0]
        # select those detections
        # print("---before indices")
        # print(boxes)
        # print(scores)
        # print(labels)
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        
        PredictionString = ''
        for i in range(len(boxes)):
            if scores[i]>0:
                x1,y1,x2,y2 = [int(x) for x in boxes[i]]
                labels_tmp = [x1,y1,x2-x1,y2-y1]
                PredictionString += " "+str(scores[i])[:3]+" "+" ".join([str(x) for x in labels_tmp])
                

                image_result = {
                        'image_id': name_dict[img_name],
                        'category_id': 1,
                        'score': float(scores[i]),
                        'bbox': [float(x) for x in labels_tmp],
                      }
                results_dict.append(image_result)
        #results.append(result)
        

        #compute
        if len(img_boxes)==0 and len(boxes)>0:
            score = 0

        elif len(boxes)==0:
            score = 0
        else:
            pre_boxes = list(np.reshape(PredictionString.strip().split(" "),(-1,5)))
            #print(img_boxes)
            #print(pre_boxes)
            score = computeScore(img_boxes, pre_boxes)
        #print(score)
        results.append(score)

        
    print("----myscore: ",np.mean(results))
        

    filepath = 'pre_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results_dict, open(filepath, 'w'), indent=4)


    #initialize COCO ground truth api
    annType = ['segm','bbox','keypoints']
    annType = annType[1]  

    dataDir='../data/wheat/train'
    annFile = '../data/wheat/instances_val.json'
    cocoGt=COCO(annFile)
    #initialize COCO detections api
    resFile='pre_results.json'
    cocoDt=cocoGt.loadRes(resFile)

    imgIds=sorted(cocoGt.getImgIds())
    imgIds=imgIds[0:100]
    imgId = imgIds[np.random.randint(100)]

    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':

    data_dict = readData()
    print(len(data_dict))
    evaluate(data_dict, data_path = '../data/wheat/train/')