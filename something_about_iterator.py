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
        return iter(self._child)  # 相当于 self._child.__iter()__ ,可迭代对象list调用这个方法返回迭代器。

# 使用生成器函数创建新的迭代模式.与普通函数不一样，生成器函数只能用于迭代操作。只响应next()操作。

def frange(start,end,increment):  # 自定义一个生成浮点数的迭代器
    x = start
    while x<end:
        yield x
        x+=increment

# 在对象上实现自定义迭代协议，最方便的方案是使用一个生成器函数，如下，对Node实现深度优先搜索遍历树形结构。
class Node_n:
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


# 反向迭代
'''
从相反的方向迭代对象，可以使用reversed()函数，但是要求对象里面的元素是已知的，需要先转换成list.
'''

# f = open('somefile')
# for line in reversed(list(f)):  # 如果元素很多的话，转成list将消耗很大的内存
#     print(line,end='')

# 可以在自定义类上，实现__reversed__()方法来反向迭代，再调用reversed函数反向迭代，就不需要将数据填充到一个列表中，再去迭代这个列表

class Count_down:
    def __init__(self,start):
        self.start = start

    def __iter__(self):   # forward iterate
        n = self.start
        while n >0:
            yield n
            n -=1

    def __reversed__(self):   # reverse iterate
        n = 1
        while n<self.start:
            yield n
            n+=1

for i in Count_down(10):
    print(i)

print('reversed countdown:'.center(30,'*'))

for i in reversed(Count_down(10)):
    print(i)

# 带有外部状态的生成器函数,将生成器函数放在自定义类的__iter__()方法中.自定义类的属性，可以做一些我们想做的事情
from collections import deque

class LineHistory:
    def __init__(self,f):
        self.lines = f
        self.history = deque(maxlen=5)

    def __iter__(self):  # 定义了__iter__()方法，那么类生成的实例就是可迭代对象，对其进行迭代操作时，就会触发这个方法，将生成器放在里面。
        for line_number, line in enumerate(self.lines,start=1):
            self.history.append((line_number,line))
            yield line

    def clear(self):
        self.history.clear()



# 同时迭代多个序列，可以使用zip()函数,返回产生一个元组的迭代器
print('enumerate'.center(30,'*'))
a = ['a','b','c']
b = [1,2,3,4]      # 两个序列的长短不一致，迭代到最短序列的数量时，迭代就会终止。
for alpha,num in zip(a,b):
    print(alpha,num)

from itertools import zip_longest   # 按照最长的序列迭代，缺少的用None填充。
for alpha,num in zip_longest(a,b):
    print(alpha,num)


# 想对多个对象执行相同的操作，但是对象又在不同的容器中时，写多个循环，不太优雅和缺少可读性，用itertools.chain(),接受一个可迭代对象列表作为输入，返回一个迭代器，有效地屏蔽掉在多个容器中迭代的细节。
from itertools import chain
a = [1,2,3]
b = ['x','y','z']

for i in chain(a,b): # chain(a,b)返回一个迭代器
    print(i)
# 如果这么写 for i in a+b: pass    那么会生成一个新的列表，会占用内存。


print('展开嵌套的序列'.center(20,'*'))
# 展开嵌套的序列。
from collections import Iterable

def flatten(items,ignore_types=(str,bytes)):   # ignore_types 防止将字符串，和字节码割裂。需要跳过迭代他们，作为整体返回
    for x in items:
        if isinstance(x,Iterable) and not isinstance(x,ignore_types):
            yield from flatten(x)    # 用 yield from 返回迭代器的子元素
        else:
            yield x

# 如果不使用yield from 那么就得多写一个for循环
# def flatten(items, ignore_types=(str, bytes)):
#     for x in items:
#         if isinstance(x, Iterable) and not isinstance(x, ignore_types):
#             for i in  flatten(x):
#                 yield i
#         else:
#             yield x

a = [1,['a',[2,3]],'c']
for i in flatten(a):
    print(i)     # 1,a,2,3,c


# 迭代器代替while循环；在需要循环调用某个函数时，往往使用while循环，然后再在循环内部定义break的条件，可以用迭代器重写这个循环.


CHUNKSIZE=8192

def process_data(data):
    pass

def reader(s):
    while True:
        data = s.recv(CHUNKSIZE)
        if data =='':
            break
        process_data(data)

# 用iter()改写上面的循环，iter函数接受一个callable对象，和一个标记值，作为输入参数，然后创建一个迭代器，不断调用callable对象，知道返回标记值，则迭代结束。
def reader2(s):
    for chunk in iter(lambda:s.recv(CHUNKSIZE),''):   # 用lambda函数，创建一个没有参数的callable对象。并且为recv函数提供size参数。
        process_data(chunk)