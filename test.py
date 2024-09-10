'''
Author: wds-dxh wdsnpshy@163.com
Date: 2024-09-01 14:18:27
LastEditors: wds-dxh wdsnpshy@163.com
LastEditTime: 2024-09-10 16:07:57
FilePath: /Drone_monitoring/test.py
Description: 
微信: 15310638214 
邮箱：wdsnpshy@163.com 
Copyright (c) 2024 by ${wds-dxh}, All Rights Reserved. 
'''
import cv2
import time
from djitellopy import Tello
from tool.process_fram import process_fram_and_say
import os
import threading

os.environ['YOLO_VERBOSE'] = str(False)  # 不打印yolov8信息
from ultralytics import YOLO

class Drone:
    def __init__(self, model):
        # tello初始化
        self.tello = Tello()
        self.tello.connect()
        time.sleep(0.5)
        self.tello.streamon()
        self.tello.takeoff()
        time.sleep(0.5)
        self.tello.move_up(100)
        time.sleep(0.5)
        self.tello.set_speed(10)
        # self.tello.flip_forward()
        time.sleep(0.5)
        battery = self.tello.get_battery()
        print('成功采集：{}，飞机电量：{}'.format(1,battery))
        # 推理模型初始化
        self.process_fram = process_fram_and_say(model)
        self.process_fram.wds_say("开始监控")

        time.sleep(2)
        self.lock = threading.Lock()  # 线程锁，保证线程安全
        self.star_jiance = True

    def async_say(self, message):
        """在单独的线程中执行播报，防止阻塞主线程"""
        with self.lock:
            self.process_fram.wds_say(message)
            print(message)
            self.star_jiance = True

    def main(self):
        start_time = time.time()  # 初始化时间，定时发送指令，防止无人机降落
        cls = None
        xywh = None

        while True:
            # 读取视频帧
            frame = self.tello.get_frame_read().frame
            height, width = frame.shape[:2]
            frame = cv2.resize(frame, (int(width / 1.5), int(height / 1.5)))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 1)  # 翻转

            # 处理视频帧
            frame, cls, xywh = self.process_fram.process_fram_ob(frame)
            # print(xywh)
            # print(cls)

            # 如果判断是溺水，就播报（多线程处理）
            if self.star_jiance:
                if cls is not None and len(cls) > 0:
                    if cls[0] == 1:
                        threading.Thread(target=self.async_say, args=("发现溺水",)).start()
                        self.star_jiance = False

            if self.star_jiance:
                if xywh is not None and len(xywh) > 0:
                    aaa = xywh[0][1] + xywh[0][3] / 2
                    if aaa < 400:
                        threading.Thread(target=self.async_say, args=("进入危险区域",)).start()
                        print("aaa",aaa)
                        self.star_jiance = False

            # 显示视频流
            cv2.imshow('demo', frame)

            status_con = 1
            # 定期发送无人机指令，避免自动降落
            if time.time() - start_time > 10 and status_con%2 == 1:
                print("上升")
                self.tello.move_up(20)
                start_time = time.time()
                status_con += 1
                time.sleep(0.01)
                
            if time.time() - start_time > 10 and status_con%2 == 0:
                print("下降")
                self.tello.move_down(20)
                start_time = time.time()
                sstatus_con += 1
                time.sleep(0.01)



            # 如果按下 'q' 键则退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.tello.land()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    model = YOLO('./models/best.pt')
    drone = Drone(model)
    drone.main()