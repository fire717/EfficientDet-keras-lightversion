import os
import cv2
import random
import numpy as np # linear algebra
import pandas as pd
import glob
#from utils import postprocess_boxes
from train import efficientdet
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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

def predict():
    print("start...")
    phi = 0
    weighted_bifpn = False
    
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    classes = {0:"wheat"}
    num_classes = len(classes)
    score_threshold = 0.1
    colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    _, model = efficientdet(phi=phi,
                            num_classes=num_classes,
                            score_threshold=score_threshold,
                            weighted_bifpn=weighted_bifpn,
                           freeze_bn=False,
                           detect_quadrangle=False)

    model_path = "models/my_model.h5"
    model.load_weights(model_path, by_name=True)
    print("finish load model")
    
    results = []
    show = True
    for image_path in glob.glob('./*.jpg'):
        image = cv2.imread(image_path)
        # BGR -> RGB
        image = image[:, :, ::-1]
        src_image = image.copy()
        h, w = image.shape[:2]
        #cv2.imwrite("pre_img1.jpg", image)
        image, scale = preprocess_image(image, image_size=image_size)
        #cv2.imwrite("pre_img2.jpg", image)
        # run network
        #start = time.time()
        boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        # print("---raw ")
        # print(boxes)
        # print(scores)
        # print(labels)

        # b
        boxes_tmp, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        #print("cost time:",time.time() - start)
        boxes = boxes_tmp.copy()
        boxes= postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

        boxes_keep_index = py_nms(boxes, scores, 0.4)
        boxes = boxes[boxes_keep_index]
        scores = scores[boxes_keep_index]
        labels = labels[boxes_keep_index]

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
                PredictionString += " "+str(scores[i])[:6]+" "+" ".join([str(x) for x in labels_tmp])
            
        result = {
            'image_id': os.path.basename(image_path),
            'PredictionString': PredictionString
        }
        results.append(result)
        
        if show:
            # print(boxes)
            # print(scores)
            # print(labels)
            draw_boxes(src_image, boxes, scores, labels, colors, classes)
            show=False
            plt.subplots(figsize = (10,10))
            plt.axis('off')
            plt.imshow(src_image)

            # save_path = os.path.join('../',os.path.basename(image_path))
            # cv2.imwrite(save_path, src_image)

    print(results[0:2])
    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    print(test_df.head())
    
    
    
    
    test_df.to_csv('submission.csv', index=False)

predict()