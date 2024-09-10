'''
Author: wds-dxh wdsnpshy@163.com
Date: 2024-05-06 18:35:54
LastEditors: wds-dxh wdsnpshy@163.com
LastEditTime: 2024-05-06 18:42:16
FilePath: /Chinese_massage/tool/get_point.py
Description: 获取关键点的xy坐标
微信: 15310638214 
邮箱：wdsnpshy@163.com 
Copyright (c) 2024 by ${wds-dxh}, All Rights Reserved. 
'''
import numpy as np

def convert_pions(pions):
    list_xy = pions.tolist()  # convert the Keypoints object to a list
    list_xy = list_xy[0]
    return list_xy
    