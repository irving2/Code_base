#!/usr/bin/env python
# coding=utf-8
# Project: Code_base
# Author : chenwen_hust@qq.com
# Date   : 2019/3/23

from collections import ChainMap


'''
用chainMap 将两个字典合并，但是并不会生成一个新的字典，在合并的子字典中修改，chainMap合并的字典也会同步更改，这是与字典的update方法最大的区别。
'''
a = {'x': 1, 'z': 3 }
b = {'y': 2, 'z': 4 }
merged = ChainMap(a,b)
print(merged)       # ChainMap({'x': 1, 'z': 3}, {'y': 2, 'z': 4})
# 对合并的字典merged操作，只会对其第一个字典有效
print(merged['z'])   # 3
del merged['x']
print(merged)        # ChainMap({'z': 3}, {'y': 2, 'z': 4})


values = ChainMap()
values['x']=1
print(values)  #ChainMap({'x': 1})

# 添加一个新的映射
values1 = values.new_child()
values1['x']=2
print(values1)  #  ChainMap({'x': 2}, {'x': 1})

# 添加一个新的字典
values2 = values1.new_child()
values2['x'] = 3
print(values2)     # ChainMap({'x': 3}, {'x': 2}, {'x': 1})

# 抛弃掉首个字典映射
values3 = values2.parents
print(values3)       # ChainMap({'x': 2}, {'x': 1})