'''
Author: wds-dxh wdsnpshy@163.com
Date: 2024-09-09 12:01:29
LastEditors: wds-dxh wdsnpshy@163.com
LastEditTime: 2024-09-10 15:58:51
FilePath: /Drone_monitoring/tool/process_fram.py
Description: 
微信: 15310638214 
邮箱：wdsnpshy@163.com 
Copyright (c) 2024 by ${wds-dxh}, All Rights Reserved. 
'''

import pyttsx3      #pip install pyttsx3 -i https://pypi.tuna.tsinghua.edu.cn/simple
import time         #pip install py3-tts -i https://pypi.tuna.tsinghua.edu.cn/simple
import cv2
import os
os.environ['YOLO_VERBOSE'] = str(False)#不打印yolov8信息
from ultralytics import YOLO
from tool import get_point
from tool.get_need_result import convert_boxes


class process_fram_and_say:
    def __init__(self,model):
        self.model = model
        self.engine = pyttsx3.init()
        self.engine.setProperty('voice', 'zh')
        self.engine.setProperty('rate', 150)
    
    def wds_say(self,say_str):     #定义一个线程函数，用于语音识别。name是检测病人的症状，从而判需要按摩的穴位
        # self.engine.say(say_str)
        # self.engine.runAndWait()
        # mac
        os.system('say ' + say_str)
        # time.sleep(2)
        print("语音识别结果：", say_str)

    def process_fram_yolo(self,frame):
        start_time = time.time()
          
        if frame is not None:
            results = self.medel.predict(frame,conf=0.1,max_det=1,save=False)
            # 在帧上可视化结果
            frame = results[0].plot()
            #绘画关键点测试，绘画关键点，6和7
            pions = results[0].keypoints.xy
            pions_list = get_point.convert_pions(pions)
            if len(pions_list) == 0:
                return frame
                
            #画出所有关键点
            # for i in range(len(pions_list)):#是浮点数，需要转换成整数
            #     cv2.circle(frame, (int(pions_list[i][0]), int(pions_list[i][1])), 5, (0, 255, 255), -1)
            #     cv2.putText(frame, str(i), (int(pions_list[i][0]), int(pions_list[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"{fps:.1f} FPS", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
    def process_fram_ob(self,frame):
        start_time = time.time()
        results = self.model.predict(frame, conf=0.6, imgsz=(640, 480), max_det=1, save=True)
        annotated_frame = results[0].plot()
        boxes = results[0].boxes
        xywh, cls = convert_boxes(boxes)
        print(cls)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"{fps:.1f} FPS", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 在h = 600的地方画一条横线
        cv2.line(annotated_frame, (0, 600), (1280, 600), (0, 0, 255), 10)
        start_time = time.time()
        return annotated_frame,cls,xywh
    
    def process_fram_mp(self,frame):
        pass
    

if __name__ == "__main__":
    model = YOLO('./models/best.pt')
    process_fram_and_say = process_fram_and_say(model)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = process_fram_and_say.process_fram_ob(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break