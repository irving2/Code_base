#!/usr/bin/env python
# coding=utf-8
# Project: Code_base
# Author : chenwen_hust@qq.com
# Date   : 2019/3/23

'''
迭代器
'''
item = [1,2,3,4]
it = iter(item)  # 构建了一个迭代器  相当于可迭代对象调用  item.__iter__()

# 可以用for循环迭代，不需要自己捕捉迭代停止的StopIteration异常
for i in it :
    print(i)

# 不用for循环，也可以这样实现
while True:
    try:
        print(next(it))
    except StopIteration:
        print('stop iterate!')
        break

# 也可以这样，迭代完成后，返回指定的内容:
it_2 = iter([4,5,6,7])
while True:
    i = next(it_2,'stop')
    if i=='stop':
        print('stop iterate!'.center(20,'*'))
        break
    print('return:',i)


# 代理迭代   在自定义容器类中定义一个__iter()__方法，将迭代操作代理到容器内部的对象上去。
class Node:
    def __init__(self,value):
        self._value = value
        self._child = []

    def __repr__(self):
        return 'Node:{}'.format(self._value)

    def add_child(self,node):
        self._child.append(node)

    def __iter__(self):
        return iter(self._child)  # 相当于 self._child.__iter()__ ,可迭代对象调用这个方法返回迭代器。

# 使用生成器函数创建新的迭代模式.与普通函数不一样，生成器函数只能用于迭代操作。只响应next()操作。

def frange(start,end,increment):  # 自定义一个生成浮点数的迭代器
    x = start
    while x<end:
        yield x
        x+=increment

# 在对象上实现自定义迭代协议，最方便的方案是使用一个生成器函数，如下，对Node实现深度优先搜索遍历树形结构。
class Node:
    def __init__(self,value):
        self._value = value
        self._child = []

    def __repr__(self):
        return 'Node:{}'.format(self._value)

    def add_child(self,node):
        self._child.append(node)

    def __iter__(self):
        return iter(self._child)  # 相当于 self._child.__iter()__ ,可迭代对象调用这个方法返回迭代器。

    def deep_first(self):
        yield self    # 先返回自己
        for c in self._child:
            yield from c.deep_first()  # 递归搜索子节点