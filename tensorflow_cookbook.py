#!/usr/bin/env python
# coding=utf-8
# Project: Code_base
# Author : chenwen_hust@qq.com
# Date   : 2019/4/3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np

# tf.nn.batch_norm_with_global_normalization()   归一化
# # tensorflow 中的数据结构tensor张量
# z = tf.zeros([3,4])
# z_ones = tf.ones([3,4])
# z_fill = tf.fill([3,4],110)
#
# z_constant = tf.constant([1,2,3])
# line_tsr = tf.linspace(start=0.0,stop=1.0,num=2)
# randunif = tf.random_uniform([3,4],minval=-1,maxval=1)  # 一定范围内的均匀分布
# randnorm = tf.random_normal([4,5],mean=0,stddev=1)
#
# tf.truncated_normal([2,3],mean=1,stddev=12)
# tf.random_shuffle()

# 用tf.Variable() 封装传入的张量，创建变量。
# 创建张量并不一定用Tensorflow 内建函数。可以用：
# tf.convert_to_tensor()

my_var = tf.Variable(tf.zeros([2,3]))
identity_matrix = tf.diag([1,2,3])
A = tf.truncated_normal([2,3])
B = tf.fill([2,3],value=5.0)
C = tf.random_uniform([2,3])
D = tf.convert_to_tensor(np.arange(6).reshape(2,3).astype(np.float32))
E = tf.add_n([A,B,C,D])
F = tf.matmul(A,tf.transpose(B))

with tf.Session() as sess:
    print(sess.run(identity_matrix))
    print(sess.run(A))   # sess.run 一个tensor后，返回一个numpy 数组。
    print(sess.run(D))
    print(sess.run(F))
    print(sess.run(tf.matrix_determinant(F)))








