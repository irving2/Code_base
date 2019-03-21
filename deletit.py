# -*- coding: utf-8 -*-
# @Time    : 2019/1/8 16:13
# @Author  : Irving


'''
pytest
'''
# def my_fun():
#     print 'this is my fun'
#
#
# def test_my_fun():
#     return
#
# test_my_fun()
#
# a={'a':1}
# print a.setdefault('b',2)
# print a
#
# print a.get('c',0)
a = {'a':1,
     'b':2,
     'c':3}

b = {'a':1,
     'w':100}

# res= {k:a[k] for k in a.keys()-b.keys()}
# print(res)

from collections import Counter
words  = ['i','a','m']
c = Counter(words)

more_words = ['m','w','e','n']
for w in more_words:
     c[w]+=1
print('end')