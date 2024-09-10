'''
Author: wds-dxh wdsnpshy@163.com
Date: 2024-09-08 22:48:09
LastEditors: wds-dxh wdsnpshy@163.com
LastEditTime: 2024-09-09 10:27:16
FilePath: /19.手势控制无人机/10_collect_vedio.py
Description: 
微信: 15310638214 
邮箱：wdsnpshy@163.com 
Copyright (c) 2024 by ${wds-dxh}, All Rights Reserved. 
'''
import cv2
import numpy as np
import time
from djitellopy import Tello

class Collect:
    def __init__(self):
        # tello初始化
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        self.tello.takeoff()
        self.tello.move_up(100)
        self.tello.set_speed(10)

    def main(self, label_id=1, duration=20):
        # 定义视频编码器和输出视频文件格式
        frame_width = 640  # 视频宽度
        frame_height = 480  # 视频高度
        # 设置视频编码器和输出文件名
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # output_filename = os.path.join('videos', 'output.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 mp4v 编码
        out = cv2.VideoWriter(f'./trainingVideos/{label_id}.avi', fourcc, 10.0, (frame_width, frame_height))

        start_time = time.time()

        while True:
            # 读取视频帧
            frame = self.tello.get_frame_read().frame
            height, width = frame.shape[:2]
            frame = cv2.resize(frame, (int(width/1.5), int(height/1.5)))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 翻转
            frame = cv2.flip(frame, 1)

            # 写入视频文件
            out.write(frame)

            # 获取电量
            battery = self.tello.get_battery()

            print('成功采集：{}，飞机电量：{}'.format(1,battery))

            # 显示视频流
            cv2.imshow('demo', frame)

            # 判断是否达到设定的采集时长
            if time.time() - start_time > duration:
                self.tello.move_up(20)
                start_time = time.time()
                print('采集15s视频')

            # 如果按下 'q' 键则退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        time.sleep(1)
        self.tello.land()        
        # 释放视频流和窗口
        out.release()
        cv2.destroyAllWindows()

# 实例化
collect = Collect()
# 准备5秒
time.sleep(5)
# 标签为1，采集20秒的视频
collect.main(1, 10)