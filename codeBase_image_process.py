# coding=utf-8
import numpy as np


def rgb2gray(rgb):
    # rgb三通道转灰度
    if len(rgb.shape) is 3:
        return np.dot(rgb[...,:3],[0.299, 0.587, 0.114])
    else:
        return rgb


if __name__ == '__main__':
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(a[:2])
