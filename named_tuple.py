#!/usr/bin/env python
# coding=utf-8
# Project: Code_base
# Author : chenwen_hust@qq.com
# Date   : 2019/3/22

'''
解决问题：有一个序列元素，如果在代码中使用索引下标取值，让代码难阅读，且序列元素的数量和顺序有更改的话，代码会出错
'''

from collections import namedtuple

Coord = namedtuple('Coordinate',['x','y','z'])

c1 = Coord(x=100,y=200,z=20)
print(c1)       # Coordinate(x=100, y=200, z=20)
print(c1.x)     # 100

# 注意 跟tuple一样，里面的元素是不能更改的；可以用实例的_replace方法更改.会生成一个新的对象
c1 = c1._replace(x=120)
print(c1)    # Coordinate(x=120, y=200, z=20)



c_prototype = Coord(None,None,None)
# 可以将字典转成namedtuple
def dict_to_coord(d):
    return c_prototype._replace(**d)   #   函数中调用全局变量时，能读取全局变量，无法对全局变量重新赋值，但是对于可变类型，可以对内部元素进行操作，比如可以append

d = {'x':1,'y':2,'z':3}

new_c = dict_to_coord(d)
print(new_c)


print(any([None,None,None]))