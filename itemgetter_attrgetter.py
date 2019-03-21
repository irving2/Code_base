#!/usr/bin/env python
# coding=utf-8
# Project: Code_base
# Author : chenwen_hust@qq.com
# Date   : 2019/3/21


# 用operator 的itemgtter排序一个序列的字典。但是如果其中某个字典没有这个key，会抛出keyError异常
# sorted的参数key要求是一个callable类型,operater.itemgetter负责从字典中产生一个用于排序的元素。也可以用lambda函数代替itemgetter，但是itemgetter运行会快一些。
# 如果是一个不能比较大小的对象实例组成的序列，使用实例的某个属性排序，可以使用operator.attrgetter.也可以使用lambda函数，但是运行速度会慢一点
import operator
data = [{'fname': 'Brian', 'lname': 'Jones', 'uid': 1003},
    {'fname': 'David', 'lname': 'Beazley','uid':1005},
    {'fname': 'John', 'lname': 'Cleese', 'uid': 1001},
    {'fname': 'Big', 'lname': 'Jones', 'uid': 1004}]

s_uid = sorted(data,key=operator.itemgetter('uid','lname'))
print(s_uid)


class Person:
    def __init__(self,name,uid):
        self.name = name
        self.uid = uid

    def __repr__(self):
        return 'Person:{}'.format(self.name)

data2 = [Person('zhangsan',10),Person('lisi',5)]
data2_sorted = sorted(data2,key=operator.attrgetter('uid'))
print(data2_sorted)