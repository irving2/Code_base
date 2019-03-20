#!/usr/bin/env python
# encoding: utf-8
# author: chenwen_hust@qq.com
# project: Code_base
# filename: dedupe.py
# time: 2019/3/20 23:15

'''
利用生成器及模仿min,max,sorted等内建函数的key参数，实现一个去除重复元素的，并且保持顺序不变的方法。如果用集合的话，不能保持顺序不变;如果处理的对象不能hash的话，也不能直接转为集合来处理
'''
a = [{'a':1},{'a':1}]
# res = set(a)  # TypeError: unhashable type: 'dict'

def dedupe(items,key=None):
    seen = set()
    for item in items:
        value = item if key is None else key(item)  # 如果为None就直接对item处理，否则使用传入的函数key来处理item
        if value not in seen:
            yield item
            seen.add(value)

if __name__ == '__main__':
    a = [{'x': 1, 'y': 2}, {'x': 1, 'y': 3}, {'x': 1, 'y': 2}, {'x': 2, 'y': 4}]
    res = dedupe(a,key=lambda k:k['x'])
    print(list(res))   # [{'x': 1, 'y': 2}, {'x': 2, 'y': 4}]

