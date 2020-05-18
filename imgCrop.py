import cv2
import json
import numpy as np
import os
import time
import glob
import sys

from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes



def getAllName(file_dir, tail_list = ['.jpg','.png']): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in tail_list:
                L.append(os.path.join(root, file))
    return L


def main(read_path, save_path):
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    phi = 0

    model_path = "checkpoints/model.h5"
    #'models/efficientdet-d0.h5'
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    # coco classes
    classes = {value['id'] - 1: value['name'] for value in json.load(open('coco_90.json', 'r')).values()}
    num_classes = 1
    score_threshold = 0.3
    #colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    colors = [[0,0,255]]
    _, model = efficientdet(phi=phi,
                            num_classes=num_classes,
                            score_threshold=score_threshold,
                            weighted_bifpn=False,
                           freeze_bn=False,
                           detect_quadrangle=False)
    model.load_weights(model_path, by_name=True)


    img_names = getAllName(read_path)
    for image_path in img_names:
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

        # select indices which have a score above the threshold
        indices = np.where(scores[:] > score_threshold)[0]

        # select those detections
        boxes = boxes[indices]
        labels = labels[indices]

        #draw_boxes(src_image, boxes, scores, labels, colors, classes)

        for i,box in enumerate(boxes):
            xmin, ymin, xmax, ymax = list(map(int, box))
            print(xmin, ymin, xmax, ymax)
            new_image = src_image[ymin:ymax, xmin:xmax]

            save_name = os.path.basename(image_path).split(".")[0]+"_box"+str(i)+"."+os.path.basename(image_path).split(".")[1]
            save_path_final = os.path.join(save_path, save_name)
            cv2.imwrite(save_path_final, new_image)



if __name__ == '__main__':

    read_path = sys.argv[1]
    save_path = sys.argv[2]
    main(read_path, save_path)
