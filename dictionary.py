#!/usr/bin/env python
# coding=utf-8
# Project: Code_base
# Author : chenwen_hust@qq.com
# Date   : 2019/3/20

'''
一些字典的操作:
1.用zip将字典的k和v反转,再使用min和max函数取得字典v的最大和最小值及对应的key.
2.对字典进行数学运算，比如min和max,都只作用于字典的key，返回的结果也是字典的key。可以利用min和max函数提供的key参数解决
3.collecions.OrderedDict  管理一个根据插入顺序管理的字典
'''

from collections import OrderedDict


price = {
    'ACME': 45.23,
    'AAPL': 612.78,
    'IBM': 205.55,
    'HPQ': 37.20,
    'FB': 10.75
}

price_zip = zip(price.values(),price.keys())
print(price_zip,type(price_zip))  # <zip object at 0x000001737D594888> <class 'zip'>
for i in price_zip:   # price_zip 为只能访问一次的迭代器
    print(i)
'''
result:
(45.23, 'ACME')
(612.78, 'AAPL')
(205.55, 'IBM')
(37.2, 'HPQ')
(10.75, 'FB')
'''
for j in price_zip:  #  不能继续迭代了。
    print('in j loop')

min_v = min(zip(price.values(),price.keys()))
max_v = max(zip(price.values(),price.keys()))
print(min_v,max_v)  # (10.75, 'FB') (612.78, 'AAPL')  但是，如果有value相同的多个k-v，这样处理只会返回其中一个。


# 2
print(min(price))  # AAPL   # 比较字典key的大小，给出结果
print(max(price))  # IBM

print(min(price,key=lambda k:price[k]))

# 3 OrderedDict
d = OrderedDict()
d['a']  =1
d['c']  =3
d['b'] = 2

for k,v in d.items():
    print(k,v)
''' output:
a 1
c 3
b 2
'''


