#!/usr/bin/env python
# coding=utf-8
# Project: Code_base
# Author : chenwen_hust@qq.com
# Date   : 2019/4/3


import csv
from collections import namedtuple

# 读写csv数据，和namedtuple结合使用.
with open('somefile.csv') as f:
    f_csv = csv.reader(f)        # 先生成一个reader对象
    head = next(f_csv)
    Row = namedtuple('Row',head)
    for r in f_csv:
        row = Row(*r)           # 可以避免使用list的索引。

# 可以将csv读取到字典，也可以方便地根据header来取到相对应的内容
with open('somefile.csv') as f:
    f_csv = csv.DictReader(f)   # 先生成一个DictReader对象
    for row in f_csv:
        ...         # 可以使用row['id']来取到每行对应的值了

# 将数据写入csv文件


headers = ['Symbol','Price','Date','Time','Change','Volume']
rows = [('AA', 39.48, '6/11/2007', '9:36am', -0.18, 181800),
         ('AIG', 71.38, '6/11/2007', '9:36am', -0.15, 195500),
         ('AXP', 62.58, '6/11/2007', '9:36am', -0.46, 935000),
       ]

with open('somefile.csv','w') as f:
    f_csv=csv.writer(f)                #先创建一个writer对象
    f_csv.writerow(headers)
    f_csv.writerow(rows)

# 如果是字典序列对象
headers = ['Symbol', 'Price', 'Date', 'Time', 'Change', 'Volume']
rows = [{'Symbol':'AA', 'Price':39.48, 'Date':'6/11/2007',
        'Time':'9:36am', 'Change':-0.18, 'Volume':181800},
        {'Symbol':'AIG', 'Price': 71.38, 'Date':'6/11/2007',
        'Time':'9:36am', 'Change':-0.15, 'Volume': 195500},
        {'Symbol':'AXP', 'Price': 62.58, 'Date':'6/11/2007',
        'Time':'9:36am', 'Change':-0.46, 'Volume': 935000},
        ]

with open('somefile.csv','w') as f:
    f_csv = csv.DictWriter(f,headers)
    f_csv.writeheader()
    f_csv.writerows(rows)


