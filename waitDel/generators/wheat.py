#coding: utf-8
"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from generators.common import Generator
import os
import numpy as np
from six import raise_from
import cv2
import pandas as pd


class WheatGenerator(Generator):
    """
    Generate data 
    """

    def __init__(
            self,
            data_dir,
            label_dir,
            label_name,
            classes={'wheat':0},
            image_extension='.jpg',
            **kwargs
    ):
        """
        Initialize a Pascal VOC data generator.

        Args:
            data_dir: the path of directory which contains ImageSets directory
            classes: class names tos id mapping
            image_extension: image filename ext
            **kwargs:
        """
        self.data_dir = data_dir
        self.classes = classes

        self.label_name_class = label_name.split(".")[0]
        self.df = pd.read_csv(os.path.join(label_dir,label_name))
        self.image_names = [x+image_extension for x in list(set(self.df['image_id'].values))]
        print("____ len image_names: ",len(self.image_names))

        # class ids to names mapping
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(WheatGenerator, self).__init__(**kwargs)

    def size(self):
        """
        Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        return len(self.classes)

    def has_label(self, label):
        """
        Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """
        Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """
        Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        path = os.path.join(self.data_dir, 'train', self.image_names[image_index])
        image = cv2.imread(path)
        h, w = image.shape[:2]
        return float(w) / float(h)

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        path = os.path.join(self.data_dir, 'train', self.image_names[image_index])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        img_name = self.image_names[image_index]

        df_one = self.df[self.df["image_id"]==img_name[:-4]]

        annotations = {'labels': np.empty((0,), dtype=np.int32),
                       'bboxes': np.empty((0, 4))}

        for idx,line in df_one.iterrows():
            annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label('wheat')]))
            
            box_str = line['bbox']
            x1,y1,w,h = [int(float(x)) for x in box_str[1:-1].split(',')]
            annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
                float(x1),
                float(y1),
                float(x1+w),
                float(y1+h),
            ]]))

        return annotations


if __name__ == '__main__':
    train_generator = WheatGenerator(
        data_dir = "/home/AlgorithmicGroup/yw/workshop/cloth_det/data/wheat",
        phi=0,
        batch_size=1,
    )
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    anchors = train_generator.anchors
    for batch_inputs, batch_targets in train_generator:
        image = batch_inputs[0][0]
        image[..., 0] *= std[0]
        image[..., 1] *= std[1]
        image[..., 2] *= std[2]
        image[..., 0] += mean[0]
        image[..., 1] += mean[1]
        image[..., 2] += mean[2]
        image *= 255.

        regression = batch_targets[0][0]
        valid_ids = np.where(regression[:, -1] == 1)[0]
        boxes = anchors[valid_ids]
        deltas = regression[valid_ids]
        class_ids = np.argmax(batch_targets[1][0][valid_ids], axis=-1)
        mean_ = [0, 0, 0, 0]
        std_ = [0.2, 0.2, 0.2, 0.2]

        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]

        x1 = boxes[:, 0] + (deltas[:, 0] * std_[0] + mean_[0]) * width
        y1 = boxes[:, 1] + (deltas[:, 1] * std_[1] + mean_[1]) * height
        x2 = boxes[:, 2] + (deltas[:, 2] * std_[2] + mean_[2]) * width
        y2 = boxes[:, 3] + (deltas[:, 3] * std_[3] + mean_[3]) * height
        for x1_, y1_, x2_, y2_, class_id in zip(x1, y1, x2, y2, class_ids):
            x1_, y1_, x2_, y2_ = int(x1_), int(y1_), int(x2_), int(y2_)
            cv2.rectangle(image, (x1_, y1_), (x2_, y2_), (0, 255, 0), 2)
            class_name = train_generator.labels[class_id]
            label = class_name
            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            cv2.rectangle(image, (x1_, y2_ - ret[1] - baseline), (x1_ + ret[0], y2_), (255, 255, 255), -1)
            cv2.putText(image, label, (x1_, y2_ - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow('image', image.astype(np.uint8)[..., ::-1])
        cv2.waitKey(0)
        # 36864, 46080, 48384, 48960, 49104
        # if first_valid_id < 36864:
        #     stride = 8
        # elif 36864 <= first_valid_id < 46080:
        #     stride = 16
        # elif 46080 <= first_valid_id < 48384:
        #     stride = 32
        # elif 48384 <= first_valid_id < 48960:
        #     stride = 64
        # else:
        #     stride = 128
        pass


