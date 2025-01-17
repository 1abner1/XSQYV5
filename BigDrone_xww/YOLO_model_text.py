import math

from Ros_Drone_final_one import Drone
import cv2
import numpy as np
import pygame
import torch
import pandas
import yolov5




# 加载本地的预训练模型和参数文件
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = yolov5.load("./last.pt")
model = model.to(device)
model.eval()  # 设置为评估模式

# 图像大小为640*480
PictureHight = 480
PictureWide = 640

drone = Drone(False)

ObjectName = "balloon"

SelectTimeFrame = 10
SelectTimeCount = SelectTimeFrame

AttackFlag = False

pygame.init()
screen = pygame.display.set_mode((640, 480))

# 开始循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    YOLO_image = drone.get_raw_left_image()

    #空则结束
    if YOLO_image is None:
        print("None")
        continue

    result = model(YOLO_image)

    # 将检测结果绘制在原始图像上
    result_img = result.render()
    result_img = result_img[0]
    # 将结果转换为OpenCV格式的图像，并显示在窗口中
    result_cv2 = np.array(result_img)[:, :, ::-1]
    cv2.imshow("Image", result_cv2)

    # 等待一段时间，接收用户输入
    key = cv2.waitKey(1)



    # 如果用户按下 q 键，则退出循环
    if key == ord('q'):
        break


