#!/usr/bin/env python
# coding=utf-8

# Project: Code_base
# Author : chenwen_hust@qq.com
# Date   : 2019/4/1
import os
import sys
from datetime import datetime
import time

'''print 函数的一些技巧'''
# 1.file参数可以重定向输出
print('hello_world_%s'%datetime.now().strftime('%Y-%m-%d %H:%M:%S'))  # hello_world_2019-04-01 09:32:58 默认的file参数是sys.stdout。

print('helle_world_%s'%datetime.now().strftime('%Y-%m-%d %H:%M:%S'),file=open('log.txt','a')) # 重定向到 'log.txt'文件中，以append模式往log.txt中写入信息，控制台上将看不到print输出信息。

# 2. sep 参数可以作为打印多个元素的分割符，比使用字符串的join()函数更加方便
row = ('a','b','c',1,2,3)
print('_'.join([str(x) for x in row]))  # 需要用str函数转换
print(*row,sep='_')


'''文件读取和写入'''
with open('somefile.txt','wb') as f:   # wt:以文本格式打开,wb：以二进制形式打开。
    line = 'hello'
    print(f.write(line.encode('utf-8')))     # 因为是以wb二进制格式打开，所以字符串需要编码。

with open('somefile.txt','rb') as f:   #
    data = f.read()
    print(data.decode('utf-8'))          # 因为是以wb模式打开，所以读取后需要解码


# with open('somefile.txt','x') as f:    # 在python3 中提供了x模式。如果打开的somefile已存在，那么就会抛出FileExistsError异常。就不可以不用 os.path.exist(dir)来实现判断文件是否存在。否则会直接覆盖掉已经存在的文件。
#     pass


import io

s1 = io.StringIO()
s2  = io.BytesIO()   # 创建类文件对象，操作字符串数据(文本字符串和二进制字符串)

s1.write('super mario')
print(s1.getvalue())
# StringIO 和 BytesIO 实例并没有正确的整数类型的文件描述符。 因此，它们不能在那些需要使用真实的系统级文件如文件，管道或者是套接字的程序中使用。
'''
import gzip,bz2

with gzip.open('somefile','rt') as f:
    data = f.read()

with bz2.open('somefile','rt') as f:
    data = f.read()
'''

# 在一个固定长度记录或者数据块集合上迭代
from functools import partial
RECORD_SIZE = 32
with open('somefile.txt','rt') as f:
    records = iter(partial(f.read, RECORD_SIZE),'')    # partial函数传入一个可调用对象，和参数。返回一个可调用对象;给iter()函数传入一个可调用对象和标志，返回一个可迭代对象，不断调用可调用对象，知道返回值为标志，则停止迭代。可以用来代替while 循环。
    for r in records:
        pass


# os.path 操作文件路径
path = '~/usr/tmp/data.csv'
print(os.path.basename(path))   # data.csv
print(os.path.dirname(path))    # ~/usr/tmp
print(os.path.expanduser(path))  # C:\Users\admin/usr/tmp/data.csv
print(os.path.islink(path))     # 判断是否为 symbolic link
print(os.path.realpath(path))   # 的到文件链接到的地址

# 获取文件的元数据
print(os.path.getsize('log.txt'))  # 获取文件大小  #1079    # 需要考虑是否拥有文件权限，否则会抛出PermissionError 异常
print(os.path.getmtime('log.txt'))  # 1554205243.5916805
print(time.ctime(os.path.getmtime('log.txt')))  # Tue Apr  2 19:41:58 2019

print(sys.getdefaultencoding())  # utf-8

print(os.listdir(b'.'))   # 有些需要处理大量文件的python程序，会碰到不能解码的文件名，可能造成代码中断；可以通过这样的方式得到字节字符串，然后操作的到的字节字符串，比如open()这个文件。





