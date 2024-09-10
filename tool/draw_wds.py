'''
Author: wds-dxh wdsnpshy@163.com
Date: 2024-04-06 16:17:54
LastEditors: wds-dxh wdsnpshy@163.com
LastEditTime: 2024-04-10 11:31:19
FilePath: /Chinese_massage/draw_wds.py
Description: 根据检测的结果画出关键点
微信: 15310638214 
邮箱：wdsnpshy@163.com 
Copyright (c) 2024 by ${wds-dxh}, All Rights Reserved. 
'''
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2#landmark_pb2是一个protobuf文件，里面定义了landmark的数据结构
import numpy as np


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image