# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd

import mmcv
import numpy as np
from cv2 import cv2

CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle']

PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
           [0, 80, 100], [0, 0, 230], [119, 11, 32]]


def parse_args():
    parser = argparse.ArgumentParser(
        description='evaluate_dataset_class_weight')
    parser.add_argument('dir', default='Image dir', type=str)
    parser.add_argument('--suffix', default='_gtFine_labelTrainIds.png', type=str)
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


class Counter():
    def __init__(self):
        self.count_all = 0
        self.count_array = np.zeros(19)

    def count_pixels(self, path):
        img = cv2.imread(path, 0)
        # print(np.max(img))

        for i in range(self.count_array.shape[0]):
            row = self.count_array[i]
            self.count_array[i] = row + np.count_nonzero(img == i)

        self.count_all += img.shape[0] * img.shape[1]

        return self.count_array

    def count(self, nproc, image_list):

        if nproc > 1:
            out = mmcv.track_parallel_progress(self.count_pixels, image_list,
                                               nproc, keep_order=True)

            self.count_array = np.sum(out, axis=0)
            # self.count_all = np.sum(out["count_all"],axis=0)
        else:
            mmcv.track_progress(self.count_pixels, image_list)

        return self.count_array, self.count_all

    # np.sum(count_array) == count_all


def weights_log(class_pixel, total_pixel):
    # w = np.log(class_pixel) / np.log(19)
    w = 1 / np.log1p(class_pixel)
    w = 19 * w / np.sum(w)
    return w


# https://github.com/openseg-group/OCNet.pytorch/issues/14
def main():
    args = parse_args()
    ## target dir scan
    original_dataset = sorted(list(
        mmcv.scandir(args.dir, recursive=True, suffix=args.suffix)
    ))

    original_dataset = [osp.join(args.dir, image) for image in original_dataset]

    count_array, count_all = Counter().count(args.nproc, original_dataset)

    print(f'count_array : {count_array}', count_all)
    print(np.sum(count_array))

    weights = weights_log(count_array, count_all)

    print(f'weights : {weights}')

    df = pd.DataFrame(data=count_array.reshape(1,-1), columns=CLASSES)
    df.plot(kind='bar')
    plt.show()


if __name__ == '__main__':
    main()
