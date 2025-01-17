import math

from Ros_Drone_final_one import Drone
import cv2
import numpy as np
import pygame
import torch
import pandas
import yolov5


# FrameRent为不同判断类型的方框应该的x大小
FrameRent = {"person":450,
             "sports ball":200}

nowy = 0
nowx = 0

MaxDis = 1
changeFlag = True

AfterPointFlag = False

arrive_point = [[-1,-1],[-1,1],[1,1],[1,-1]]
NearestPoint = None

def FourPointSearch(position,drone:Drone):
    global NearestPoint,arrive_point,AfterPointFlag
    if NearestPoint is None:
        dis = 5000
        for point in arrive_point:
            newdis = math.sqrt(pow((position[0]-point[0]),2)+pow((position[1]-point[1]),2))
            if newdis<dis:
                dis=newdis
                NearestPoint = point
    if math.fabs(position[0] - NearestPoint[0])<0.05:
        nowy = 0
    elif position[0] - NearestPoint[0]>0:
        nowy = -0.15
    else:
        nowy = 0.15

    if math.fabs(position[1] - NearestPoint[1])<0.05:
        nowx = 0
    elif position[1] - NearestPoint[1]>0:
        nowx = -0.15
    else:
        nowx = 0.15

    if nowy==0 and nowx==0:
        AfterPointFlag = True
    drone.MoveList(position=[nowy, nowx, 0], is_world_position=True)

def FourPointAfter(position,drone):
    global NearestPoint
    if NearestPoint == [-1,-1]:
        if math.sqrt(pow((position[0]-NearestPoint[0]),2)+pow((position[1]-NearestPoint[1]),2))<0.05:
            NearestPoint = [-1,1]
        else:
            nowy = -0.15
            drone.MoveList(position=[nowy, 0, 0], is_world_position=True)
    elif NearestPoint == [-1,1]:
        if math.sqrt(pow((position[0]-NearestPoint[0]),2)+pow((position[1]-NearestPoint[1]),2))<0.05:
            NearestPoint = [1,1]
        else:
            nowx = 0.15
            drone.MoveList(position=[0, nowx, 0], is_world_position=True)
    elif NearestPoint == [1,1]:
        if math.sqrt(pow((position[0]-NearestPoint[0]),2)+pow((position[1]-NearestPoint[1]),2))<0.05:
            NearestPoint = [1,-1]
        else:
            nowy = 0.15
            drone.MoveList(position=[nowy, 0, 0], is_world_position=True)
    elif NearestPoint == [1,-1]:
        if math.sqrt(pow((position[0]-NearestPoint[0]),2)+pow((position[1]-NearestPoint[1]),2))<0.05:
            NearestPoint = [-1,-1]
        else:
            nowx = -0.15
            drone.MoveList(position=[0, nowx, 0], is_world_position=True)

def TrackWithoutMove():
    global nowy,nowx,changeFlag,NearestPoint,AfterPointFlag
    # 加载本地的预训练模型和参数文件
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = yolov5.load("./UAVtoUGV/weights/best.pt")
    model = model.to(device)
    model.eval()  # 设置为评估模式

    drone = Drone(True)

    pygame.init()
    screen = pygame.display.set_mode((640, 480))

    # 图像大小为640*480
    PictureHight = 480
    PictureWide = 640

    AllowDistance = 30

    # 开始循环
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        # 判断每帧的图像，如果错误直接报错

        YOLO_image = drone.get_raw_YOLO_image()

        if YOLO_image is None:
            print("None")
            continue

        result = model(YOLO_image)

        # 将检测结果绘制在原始图像上
        result_img = result.render()

        result_img = result_img[0]
        # 将结果转换为OpenCV格式的图像，并显示在窗口中
        result_cv2 = np.array(result_img)[:, :, ::-1]

        # result_cv2 = cv2.cvtColor(result_cv2, cv2.COLOR_BGR2RGB)

        cv2.imshow("Image", result_cv2)

        # 等待一段时间，接收用户输入
        key = cv2.waitKey(1)

        answer = result.pandas().xyxy[0]
        # FootballName = "sports ball"
        ObjectName = "Y-UGV"
        Answer = answer[answer["name"] == ObjectName]

        if Answer.empty:
            print("No Target")
            # newy = nowy
            # newx = nowx

            # position,rotation = drone.get_global_pose()
            # if (not changeFlag) and (not AfterPointFlag):
            #     FourPointSearch(position,drone)
            #
            # elif (not changeFlag) and AfterPointFlag:
            #     FourPointAfter(position,drone)
            #
            # elif (math.fabs(position[0])>MaxDis or math.fabs(position[1])>MaxDis) and changeFlag:
            #     changeFlag = False
            #
            # # elif math.fabs(position[0])>MaxDis:
            # #    newy = 0
            # # elif math.fabs(position[1]) > MaxDis:
            # #    newx = 0
            # else:
            #     drone.MoveList(position=[nowy, nowx, 0], is_world_position=True)
            continue

        changeFlag = True
        AfterPointFlag = False
        NearestPoint = None
        NearAnswer = found_nearest_box(Answer)
        BallCentre_x = (NearAnswer['xmax'] - NearAnswer['xmin']) / 2 + NearAnswer['xmin']
        BallCentre_y = (NearAnswer['ymax'] - NearAnswer['ymin']) / 2 + NearAnswer['ymin']

        print("BallCentre_x",BallCentre_x)
        print("BallCentre_y",BallCentre_y)

        if math.fabs(PictureHight/2 - BallCentre_y) < AllowDistance:
            y = 0
        elif PictureHight/2 - BallCentre_y <0:
            y = -0.15
        else:
            y = 0.15

        if math.fabs(PictureWide / 2 - BallCentre_x) < AllowDistance:
            x = 0
        elif PictureWide / 2 - BallCentre_x < 0:
            x = -0.15
        else:
            x = 0.15

        nowy = y
        nowx = x
        # 方向相反
        drone.MoveList(position=[y, x, 0], is_world_position=True)

        # 如果用户按下 q 键，则退出循环
        if key == ord('q'):
            break


def found_nearest_box(answers: pandas.DataFrame):
    area = 0
    max_index = 0
    for i in list(answers.index):
        new_area = (answers['xmax'][i] - answers['xmin'][i]) *(answers['ymax'][i] - answers['ymin'][i])
        if new_area > area:
            area = new_area
            max_index = i
    # print(answers.loc[max_index])
    return answers.loc[max_index]

TrackWithoutMove()
