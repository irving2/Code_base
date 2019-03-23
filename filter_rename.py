#!/usr/bin/env python
# coding=utf-8
# Project: Code_base
# Author : chenwen_hust@qq.com
# Date   : 2019/3/22


'''
过滤序列元素
filter函数可以用于根据传入的callable的对象，产生bool值，返回序列中对应值为true的。返回一个迭代器
'''

a= [-100,1,2,3,4,5]

bool_a = [x>3 for x in a]
print(list(filter(lambda x:x>3,a)))


'''
itertools.compress也可以做这个事情，传入的是一个bool序列，同样的也是返回一个迭代器
'''
from itertools import compress

print(list(compress(a,bool_a)))

