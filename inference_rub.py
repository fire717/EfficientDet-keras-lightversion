#coding:utf-8
import cv2
import json
import numpy as np
import os
import time
import glob

from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes

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
        # print("-----------------")
        # print("order[1:]: ",order[1:])
        # print("x1[order[1:]]: ",x1[order[1:]])
        # print("np.maximum(x1[i], x1[order[1:]]): ", np.maximum(x1[i], x1[order[1:]]))
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

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    phi = 0
    weighted_bifpn = False
    model_path = "checkpoints/2020-04-21/pascal_50_0.2337_0.6807.h5"
    #'models/efficientdet-d0.h5'
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    # coco classes
    #classes = {value['id'] - 1: value['name'] for value in json.load(open('coco_90.json', 'r')).values()}
    #num_classes = 1

    with open("../data/trainval/train_classes_labeled.txt","r",encoding="utf-8") as f:
        lines = f.readlines()
    classes = {int(value.split(",")[0]):value.split(",")[0] for value in lines}
    num_classes = len(classes)

    score_threshold = 0.3
    colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    #colors = [[0,0,255]]
    _, model = efficientdet(phi=phi,
                            num_classes=num_classes,
                            score_threshold=score_threshold,
                            weighted_bifpn=weighted_bifpn,
                           freeze_bn=False,
                           detect_quadrangle=False)
    model.load_weights(model_path, by_name=True)

    for image_path in glob.glob('datasets/rub/*.jpg'):
        image = cv2.imread(image_path)
        src_image = image.copy()
        # BGR -> RGB
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        print("cost time:",time.time() - start)
        boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

        #print(boxes, scores)
        boxes_keep_index = py_nms(boxes, scores, 0.7)
        boxes = boxes[boxes_keep_index]
        scores = scores[boxes_keep_index]
        labels = labels[boxes_keep_index]

        # select indices which have a score above the threshold
        indices = np.where(scores[:] > score_threshold)[0]
        # select those detections
        boxes = boxes[indices]
        labels = labels[indices]

        draw_boxes(src_image, boxes, scores, labels, colors, classes)

        save_path = os.path.join('datasets/res/',os.path.basename(image_path))
        cv2.imwrite(save_path, src_image)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image', src_image)
        # cv2.waitKey(0)


if __name__ == '__main__':
    main()
