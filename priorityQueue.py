#!/usr/bin/env python
# encoding: utf-8
# author: chenwen_hust@qq.com
# project: Code_base
# filename: priorityQueue.py
# time: 2019/3/19 22:32


'''
利用堆heapq实现一个优先级队列，每次pop取出优先级最高的元素
'''
import heapq

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0  # 当priority相同时,可以按照入堆先后顺序不同比较

    def push(self,item,priority):
        heapq.heappush(self._queue,(-priority,self._index,item))
        self._index+=1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def __repr__(self):
        return 'PriorityQueue instance:%s'%self._queue

'''
tuple之间的比较大小运算,先拿第一个元素比较，如果相等就拿第二个元素比较，以此类推。这也是self._index的作用，当优先级相同时会拿self._index来比较。如果没有self._index，不能直接拿foo实例来比较，python3会抛出错误。但是python2.7不会报错。
'''

# heapq.merge 可以实现两个序列的合并排序。heap.merge返回一个迭代器，意味着可以在非常长的序列中使用它，而不会有太大的开销。
a = [1,3,4,9]
b = [2,5,8]
for i in heapq.merge(a,b):
    print(i)
# 注意  在使用heapq.merge前要求，序列a,b,c等都是要提前排序好。heapq.merge 不会对输入做任何的排序检测，只检查每个序列的开始部分，并返回最小的那个，这个过程一直持续到所有序列的元素都被遍历完。


if __name__ == '__main__':
    class Foo:
        pass

    f1 = Foo()
    f2 = Foo()

    pq =PriorityQueue()
    pq.push(f1,5)
    pq.push(f2,2)
    pq.push({'bar':'foo'},10)










