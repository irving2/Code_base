#!/usr/bin/env python
# coding=utf-8
# Project: Code_base
# Author : chenwen_hust@qq.com
# Date   : 2019/3/23

import random

'''
random 模块通过Mersenne Twister算法来计算生成随机数。是一个确定性的算法，可以通过random.seed()函数修改初始化种子。random模块中的函数不应该用在和密码学相关的程序中,可以用ssl模块生成随机数，
'''
random.seed(42)

# 从一个序列里面，随机选择一个元素
a = [1,2,3,4,5,6]
res = random.choice(a)
print(res)

# 从一个序列里面，随机选择多个元素
print(random.sample(a,k=3))

# 只是想将序列中的元素，随机打乱
random.shuffle(a)
print(a)            # [5, 3, 2, 6, 1, 4]

# 生成0~10之间的随机整数,包括10.
random.randint(0, 1)

# 随机获得一个3位数(二进制)的整数
random.getrandbits(k=3)  #6

# 根据一些分布函数来生成随机数
random.uniform()   # 均匀分布
random.gauss()     # 正态分布
