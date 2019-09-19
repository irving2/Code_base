#!/usr/bin/env python
# coding=utf-8
# Author : chenwen_hust@qq.com
# datetime:19-9-4 上午9:14
# project: Code_base


# 卷积层到全连接层，匹配一维向量输入和输出的维度
def num_flat_feature(x):
    size = x.size()[1:]   # 除去batch_size
    num_features = 1
    for i in size:
        num_features *= i
    return num_features


