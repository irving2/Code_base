#!/usr/bin/env python
# coding=utf-8
# Project: Code_base
# Author : chenwen_hust@qq.com
# Date   : 2019/3/23

# 在浮点数计算时，如果对精度要求非常高，比如在金融领域，不允许很小的误差传播，可以使用decimal模块进行浮点数计算。
# import decimal

a = 4.2
b = 2.1
print(a+b)   # 6.300000000000001

from decimal import Decimal

a = Decimal('4.2')
b = Decimal('2.1')
print(a+b)   # 6.3


# decimal 模块可以控制计算的各个方面，包括计算的位数和四舍五入运算。可以创建一个上下文并更改它的设置。
from decimal import localcontext

a = Decimal('1.3')
b = Decimal('1.7')

print(a/b)   # 0.7647058823529411764705882353
with localcontext() as ctx:
    ctx.prec=5
    print(a/b)   # 0.76471
