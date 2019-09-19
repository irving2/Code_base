# -*- coding: utf-8 -*-
# @Time    : 2019/1/8 11:19
# @Author  : Irving


class Counter:
    Count = 0   # This represents the count of objects of this class

    def __init__(self, name):
        self.name = name
        print(name, 'created')
        Counter.Count += 1

    def __del__(self):
        print(self.name, 'deleted')
        Counter.Count -= 1
        if Counter.Count == 0:
            print ('Last Counter object deleted')
        else:
            print (Counter.Count, 'Counter objects remaining')

x = Counter("First")
del x # 如果没有这个语句，程序会抛出Exception AttributeError
'''
程序在运行的最后调用__del__来回收，但是哪个时候，__del__里面的外部引用可能已经被销毁了，__del__() methods should do the absolute minimum needed to maintain external invariants.
有两种解决办法：
1.显式调用一些结束的方法；例如对于打开的文件对象，调用close()方法
2.使用 weak reference。弱引用，一个对象若只被弱引用所引用，则可能在任何时刻被回收。弱引用的主要作用就是减少循环引用，减少内存中不必要的对象存在的数量
    WeakValueDictionary是一个字典，字典的值是对象的弱引用，当这些值引用的对象不再被其他非弱引用对象引用时，那么这些引用的对象就可以通过垃圾回收器进行回收
'''

from weakref import WeakValueDictionary

class Counter:
    _instances = WeakValueDictionary()
    @property
    def Count(self):
        return len(self._instances)

    def __init__(self, name):
        self.name = name
        self._instances[id(self)] = self
        print (name, 'created')

    def __del__(self):
        print (self.name, 'deleted')
        if self.Count == 0:
            print ('Last Counter object deleted')
        else:
            print (self.Count, 'Counter objects remaining')

x = Counter("First")



'''
对于弱引用用于循环引用防止内存泄漏，参考： https://segmentfault.com/a/1190000005729873
'''
