#!/usr/bin/env python
# coding=utf-8
# Project: Code_base
# Author : chenwen_hust@qq.com
# Date   : 2019/3/23

from datetime import datetime,timedelta,date
import calendar

t = datetime(2019,3,23)
t = t+timedelta(days=3)
print(t)  # 2019-03-26 00:00:00

print(datetime.now())   # 2019-03-23 15:04:11.589059
print(datetime.today())   # 2019-03-23 15:04:11.589059

print(date.today())       # 2019-03-23

today = date.today()
print(calendar.monthrange(today.year,today.month))   # (4, 31)  4个星期，31天

