# -*- coding: utf-8 -*-
# @Time    : 2019/1/8 21:44
# @Author  : Irving



class Messenger:
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

m = Messenger(info="some information", b=['a', 'list'])
m.more = 11
print (m.info, m.b, m.more)
print (m.__dict__)


*a,b ='hello'
print(b)