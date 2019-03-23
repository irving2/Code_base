#!/usr/bin/env python
# coding=utf-8
# Project: Code_base
# Author : chenwen_hust@qq.com
# Date   : 2019/3/22

'''
对于一个序列的数据，当需要先过滤或转换，然后再进行计算时，可以考虑将迭代器作为参数传入函数
'''
nums = [1,2,3,4,5]

# 计算平方和
s = sum(x*x for x in nums)  # 生成器表示式作为参数， 不需要给生成器额外加个括号了，这样比较优雅。

# 将元组连接成csv
s = ('name','chenwen','birthday','111')
line = ','.join(str(x) for x in s)


# 从字典里面某个value,求最大，最小,排序,除了用itemgetter外，也可以将一个生成器传入min\max\sort这些可以接受iterable作为参数的函数中。
portfolio = [
    {'name':'GOOG', 'shares': 50},
    {'name':'YHOO', 'shares': 75},
    {'name':'AOL', 'shares': 20},
    {'name':'SCOX', 'shares': 65}
]

max_element = max(x['shares'] for x in portfolio)  # 75

import operator
max_dict = max(portfolio,key=operator.itemgetter('shares'))  # {'name': 'YHOO', 'shares': 75}
