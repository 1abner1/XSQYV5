import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import math



class Drone:
    def __init__(self, have_YOLO_camera = False):
        self.position_ = [0, 0, 0]
        self.orientation_ = [0, 0, 0, 0]

        rospy.init_node('drone_controller', anonymous=True)
        # 用来记录无人机当前位置和旋转角信息的订阅者（世界坐标）
        self.mavposeSub_ = rospy.Subscriber("/mavros/local_position/odom", Odometry, self._mavposeCallback)
        # 用来发布无人机前进命令的发布者（世界坐标）
        self.goalPub_ = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        # self.takeoffPub_ = rospy.Publisher("/control", control, queue_size=1)

        self.bridge = CvBridge()

        # 以下为左摄像头的回调函数
        self.cv_image_left = None
        self.image_left_info = [0, 0]
        self.image_left_sub = rospy.Subscriber("/stereo_cam/left/image_raw", Image, self._left_image_callback)

        # 以下为右摄像头的回调函数
        self.cv_image_right = None
        self.image_right_info = [0, 0]
        self.image_right_sub = rospy.Subscriber("/stereo_cam/right/image_raw", Image, self._right_image_callback)

        # 以下为深度摄像头的回调函数
        self.cv_image_depth = None
        self.image_depth_info = [0, 0]
        self.image_depth_sub = rospy.Subscriber("/stereo_cam/depth", Image, self._depth_image_callback)

        # 以下为拥有的彩色摄像头
        self.have_YOLO_camera = have_YOLO_camera
        if have_YOLO_camera:
            self.cv_image_YOLO = None
            self.image_YOLO_info = [0,0,0]
            self.image_YOLO_sub = rospy.Subscriber("/YOLO_camera/Image", Image, self._YOLO_image_callback)

        print("Drone Initialization Final.")



    def MoveList(self, position=None, orientation=None, is_world_position=False, is_abs_position=False):
        """
        This function can control Drone to move with List Position.
        Args:
            :param position:Size should be 3.The position is world position
            :type position: List or ndarray
            :param orientation:Size should be 4. The orientation is world orientation
            :type orientation:List or ndarray
            :return:None
        """
        goal_ = PoseStamped()
        goal_.header.stamp = rospy.Time.now()
        goal_.header.frame_id = "world"
        # 输入本来的旋转方向
        goal_.pose.orientation.w = self.orientation_[0]
        goal_.pose.orientation.x = self.orientation_[1]
        goal_.pose.orientation.y = self.orientation_[2]
        goal_.pose.orientation.z = self.orientation_[3]
        # 输入原来的目标点位置
        goal_.pose.position.x = self.position_[0]
        goal_.pose.position.y = self.position_[1]
        goal_.pose.position.z = self.position_[2]
        if position is not None:
            # 小于3无法行动
            if isinstance(position,list):
                if len(position) <3:
                    print("Position size is less than 3, can't move.")
                else:
                    # 大于3提供警告信息
                    if len(position)>3:
                        print("Position size is more than 3.Move may have problem.")
                    if is_abs_position and is_world_position:
                        goal_.pose.position.x = position[0]
                        goal_.pose.position.y = position[1]
                        goal_.pose.position.z = position[2]
                    elif is_world_position:
                        goal_.pose.position.x += position[0]
                        goal_.pose.position.y += position[1]
                        goal_.pose.position.z += position[2]
                    else:
                        if is_abs_position:
                            print("When is_world_position is False,is_abs_position should not be True,it is meaningless!")
                        # 转换成实际的world坐标
                        now_rotation_x = math.atan2(2. * (self.orientation_[0]*self.orientation_[3] + self.orientation_[1]*
                                                        self.orientation_[2]),
                                                  1. - 2. * (self.orientation_[2]*self.orientation_[2] +
                                                             self.orientation_[3]*self.orientation_[3]))
                        actual_x = position[0] * math.cos(now_rotation_x)
                        actual_y = position[0] * math.sin(now_rotation_x)
                        now_rotation_y = now_rotation_x + math.pi/2
                        actual_x += position[1] * math.cos(now_rotation_y)
                        actual_y += position[1] * math.cos(now_rotation_y)
                        goal_.pose.position.x += actual_x
                        goal_.pose.position.y += actual_y
                        goal_.pose.position.z += position[2]

            elif isinstance(position, float) or isinstance(position, int):
                if is_world_position:
                    print("Position is int or float, Position cannot be world position. It will not move.")
                else:
                    now_rotation_x = math.atan2(
                        2. * (self.orientation_[0] * self.orientation_[3] + self.orientation_[1] *
                              self.orientation_[2]),
                        1. - 2. * (self.orientation_[2] * self.orientation_[2] +
                                   self.orientation_[3] * self.orientation_[3]))
                    actual_x = position * math.cos(now_rotation_x)
                    actual_y = position * math.sin(now_rotation_x)
                    goal_.pose.position.x += actual_x
                    goal_.pose.position.y += actual_y


        # 如果有旋转输入，则改变旋转方向
        if orientation is not None:
            # 不是int无法旋转
            '''
            if not isinstance(orientation, float):
                print("Orientation size is not int, can't rotate.")
            
            else:
            '''
            if isinstance(orientation, list):
                if is_abs_position and is_world_position:
                    goal_.pose.orientation.w = orientationp[0]
                    goal_.pose.orientation.x = orientationp[1]
                    goal_.pose.orientation.y = orientationp[2]
                    goal_.pose.orientation.y = orientationp[3]
                else:
                    goal_.pose.orientation.w += orientationp[0]
                    goal_.pose.orientation.x += orientationp[1]
                    goal_.pose.orientation.y += orientationp[2]
                    goal_.pose.orientation.y += orientationp[3]
            elif isinstance(orientation, float) or isinstance(orientation, int):
                q_new = self._rotate_drone(orientation)
                goal_.pose.orientation.w = q_new[0]
                goal_.pose.orientation.x = q_new[1]
                goal_.pose.orientation.y = q_new[2]
                goal_.pose.orientation.z = q_new[3]

        # 发布执行
        print(goal_)
        self.goalPub_.publish(goal_)


    def MoveInt(self, position=None, height=None, orientation=None):
        if isinstance(position, list) or isinstance(height, list):
            print("List Position should use MoveList function!")
            return

        goal_ = PoseStamped()
        goal_.header.stamp = rospy.Time.now()
        goal_.header.frame_id = "world"

        # 输入原来的目标点位置
        goal_.pose.position.x = self.position_[0]
        goal_.pose.position.y = self.position_[1]
        goal_.pose.position.z = self.position_[2]

        # 输入本来的旋转方向
        goal_.pose.orientation.w = self.orientation_[0]
        goal_.pose.orientation.x = self.orientation_[1]
        goal_.pose.orientation.y = self.orientation_[2]
        goal_.pose.orientation.z = self.orientation_[3]

        if position is not None:
            now_rotation_x = math.atan2(
                2. * (self.orientation_[0] * self.orientation_[3] + self.orientation_[1] *
                      self.orientation_[2]),
                1. - 2. * (self.orientation_[2] * self.orientation_[2] +
                           self.orientation_[3] * self.orientation_[3]))

            actual_x = position * math.cos(now_rotation_x)
            actual_y = position * math.sin(now_rotation_x)

            goal_.pose.position.x += actual_x
            goal_.pose.position.y += actual_y

        if height is not None:
            goal_.pose.position.z += height

        # 如果有旋转输入，则改变旋转方向
        if orientation is not None:
            # 不是int无法旋转
            '''
            if not isinstance(orientation, float):
                print("Orientation size is not int, can't rotate.")

            else:
            '''

            q_new = self._rotate_drone(orientation)
            goal_.pose.orientation.w = q_new[0]
            goal_.pose.orientation.x = q_new[1]
            goal_.pose.orientation.y = q_new[2]
            goal_.pose.orientation.z = q_new[3]

        # 发布执行
        print(goal_)
        self.goalPub_.publish(goal_)



    # 深度摄像头的回调函数
    def get_raw_depth_image(self):
        """
        Get the depth camera data
        :return: Depth camera data
        :rtype: np.ndarray
        """
        return self.cv_image_depth

    def _depth_image_callback(self, data):
        """
        Callback function for the depth camera. Stores data from the depth camera
        :param data: Type is sensor_msgs.Image, no need to fill in.
        :return: None
        """
        try:
            self.cv_image_depth = self.bridge.imgmsg_to_cv2(data, "32FC1")
            self.image_depth_info[0] = data.height
            self.image_depth_info[1] = data.width
        except CvBridgeError as e:
            print("Depth Error")
            print(e)

    # 左摄像头回调函数
    def get_raw_left_image(self):
        """
        Get the left camera data
        :return: Left camera data
        :rtype: np.ndarray
        """
        return self.cv_image_left

    def _left_image_callback(self, data):
        """
        Callback function for the left camera. Stores data from the left camera
        :param data: Type is sensor_msgs.Image, no need to fill in.
        """
        try:

            self.cv_image_left = self.bridge.imgmsg_to_cv2(data, "mono8")
            self.image_left_info[0] = data.height
            self.image_left_info[1] = data.width
        except CvBridgeError as e:
            print(e)

    def show_image(self, camera_index):
        """
        Show the specific image
        :param camera_index: The camera index
        :type camera_index:String
        :return: None
        """
        if camera_index == "left":
            if self.cv_image_left is not None:
                cv2.imshow("Image window", self.cv_image_left)
                cv2.waitKey(1)
            else:
                print("image is None")

        elif camera_index == "right":
            if self.cv_image_right is not None:
                cv2.imshow("Image window", self.cv_image_right)
                cv2.waitKey(1)
            else:
                print("image is None")

        elif camera_index == "depth":
            if self.cv_image_depth is not None:
                cv2.imshow("Image window", self.cv_image_depth)
                cv2.waitKey(1)
            else:
                print("image is None")

        elif camera_index == "YOLO":
            if self.have_YOLO_camera:
                frame = self.get_raw_YOLO_image()
                if frame is not None:
                    cv2.imshow("Image window", self.cv_image_YOLO)
                    cv2.waitKey(1)
                else:
                    print("Frame is None")
            else:
                print("There is no YOLO camera when starting setting.")

        else:
            print("Camera index is not in ['left', 'right', 'depth', 'YOLO']")

    # 右摄像头的代码

    def get_raw_right_image(self):
        """
        Get the right camera data
        :return: Right camera data
        :rtype: np.ndarray
        """
        return self.cv_image_right

    def _right_image_callback(self, data):
        """
        Callback function for the right camera. Stores data from the right camera
        :param data: Type is sensor_msgs.Image, no need to fill in.
        """

        try:
            self.cv_image_right = self.bridge.imgmsg_to_cv2(data, "mono8")
            self.image_right_info[0] = data.height
            self.image_right_info[1] = data.width
        except CvBridgeError as e:
            print(e)


    # Yolo外置摄像头代码
    def _YOLO_image_callback(self, data):
        """
        Callback function for the YOLO camera. Stores data from the YOLO camera
        :param data: Type is sensor_msgs.Image, no need to fill in.
        :return: None
        """
        try:
            self.cv_image_YOLO = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.image_YOLO_info[0] = data.height
            self.image_YOLO_info[1] = data.width
        except CvBridgeError as e:
            print("YOLO Error")
            print(e)

    def get_raw_YOLO_image(self):
        """
        Get the YOLO camera data
        :return: YOLO camera data
        :rtype: np.ndarray
        """
        return self.cv_image_YOLO

    def get_global_pose(self):
        """
        Get the Drone world position
        :return: Drone world position
        """
        return self.position_, self.orientation_


    def get_camera_info(self, camera_index):
        """
        Get the specific camera info(height, width)
        :param camera_index: The camera index
        :return: The height and with of the specific camera
        :rtype:List, Size is 2
        """
        if camera_index == "left":
            if self.image_left_info[0] != 0:
                return self.image_left_info
            else:
                print("Camera Info is None now.")
        elif camera_index == "right":
            if self.image_right_info[0] != 0:
                return self.image_right_info
            else:
                print("Camera Info is None now.")
        elif camera_index == "depth":
            if self.image_depth_info[0] != 0:
                return self.image_depth_info
            else:
                print("Camera Info is None now.")
        elif camera_index == "YOLO":
            if self.have_YOLO_camera:
                if self.image_YOLO_info[0] != 0:
                    return self.image_YOLO_info
                else:
                    print("Camera Info is None now.")
            else:
                print("There is no YOLO camera when starting setting.")
        else:
            print("Camera index is not in ['left', 'right', 'depth', 'YOLO]")

    def spin(self):
        while not rospy.is_shutdown():
            pass

    def get_raw_image(self, camera_index):
        """
        Get the specific camera data
        :param camera_index: The camera index
        :type camera_index:String
        :return: Specific camera data
        :rtype: np.ndarray
        """

        if camera_index == "left":
            return self.get_raw_left_image()
        elif camera_index == "right":
            return self.get_raw_right_image()
        elif camera_index == "depth":
            return self.get_raw_depth_image()
        elif camera_index == "YOLO":
            return self.get_raw_YOLO_image()

        else:
            print("Camera index is not in ['left', 'right', 'depth', 'YOLO]")

# 回调函数，用来实时存储当前无人机的位置和旋转角
    def _mavposeCallback(self, msg):

        # 存储位置
        self.position_[0] = msg.pose.pose.position.x
        self.position_[1] = msg.pose.pose.position.y
        self.position_[2] = msg.pose.pose.position.z
        # 存储旋转角
        self.orientation_[0] = msg.pose.pose.orientation.w
        self.orientation_[1] = msg.pose.pose.orientation.x
        self.orientation_[2] = msg.pose.pose.orientation.y
        self.orientation_[3] = msg.pose.pose.orientation.z

    def _rotate_drone(self, rotation_angle):
        # 获取当前旋转四元数
        q_cur = self.orientation_

        # 计算要旋转的四元数
        rotation_vector = [0, 0, 1]
        rotation_quat = self._quaternion_about_axis(rotation_angle, rotation_vector)

        # 进行四元数插值，得到新的旋转四元数
        q_new = self._quaternion_multiply(q_cur, rotation_quat)

        # print("q_new:", q_new)

        return q_new

    def _quaternion_multiply(self,q1, q2):
        if len(q1) != 4 or len(q2) != 4:
            raise ValueError("Quaternion lists must have length 4")

        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return [w, x, y, z]

    def _quaternion_about_axis(self, angle, axis):
        # 计算四元数
        s = np.sin(np.deg2rad(angle / 2.0))
        c = np.cos(np.deg2rad(angle / 2.0))
        return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s])

    @staticmethod
    def quaternion_to_yaw(quaternion):
        # 将四元数转换为旋转矩阵
        rotation_matrix = np.array([
            [1 - 2 * (quaternion[2] ** 2 + quaternion[3] ** 2),
             2 * (quaternion[1] * quaternion[2] - quaternion[0] * quaternion[3]),
             2 * (quaternion[1] * quaternion[3] + quaternion[0] * quaternion[2])],
            [2 * (quaternion[1] * quaternion[2] + quaternion[0] * quaternion[3]),
             1 - 2 * (quaternion[1] ** 2 + quaternion[3] ** 2),
             2 * (quaternion[2] * quaternion[3] - quaternion[0] * quaternion[1])],
            [2 * (quaternion[1] * quaternion[3] - quaternion[0] * quaternion[2]),
             2 * (quaternion[2] * quaternion[3] + quaternion[0] * quaternion[1]),
             1 - 2 * (quaternion[1] ** 2 + quaternion[2] ** 2)]
        ])

        # 计算旋转矩阵的yaw角
        yaw_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        # 将弧度转换为角度
        yaw_deg = np.degrees(yaw_rad)

        return yaw_deg

    @staticmethod
    def yaw_to_quaternion(yaw_degrees):
        # 将角度转换为弧度
        yaw_radians = np.radians(yaw_degrees)

        # 计算四元数的元素
        w = np.cos(yaw_radians / 2)
        x = 0.0
        y = 0.0
        z = np.sin(yaw_radians / 2)

        # 返回四元数
        quaternion = np.array([w, x, y, z])
        return quaternion

# 测试函数用来退出的异常类
class EndException(Exception):
    pass

def drone_test():
    import pygame
    print("=================================\n"
          "=== Welcome to test the Drone ===\n"
          "=================================\n")

    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    Drone_test = Drone()



    try:
        waiting = True
        Running = True
        # 左摄像头测试
        print(
              "First we will test the camera\n"
              "You can always press p to continue and q to the next test in the pygame window\n"
              "The first camera is the left camera\n"
              "Get left camera function is class_name.get_raw_left_image()\n"
              "(Press p to continue):")


        # 一直到按下p否则不动
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        waiting = False
                        break
                elif event.type == pygame.QUIT:
                    raise EndException

        while Running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        Running = False
                        break
                elif event.type == pygame.QUIT:
                    raise EndException
            image_data_left = Drone_test.get_raw_left_image()
            print("image_data_left:", image_data_left)
            Drone_test.show_image("left")


        # 右摄像头测试
        waiting = True
        Running = True
        print(
                "Next we will test the right camera\n"
                "You can always press p to continue and q to the next test in the pygame window\n"
                "Get right camera function is class_name.get_raw_right_image()\n"
                "(Press p to continue):")

        # 一直到按下p否则不动
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        waiting = False
                        break
                elif event.type == pygame.QUIT:
                    raise EndException

        while Running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        Running = False
                        break
                elif event.type == pygame.QUIT:
                    raise EndException
            image_data_right = Drone_test.get_raw_right_image()
            print("image_data_right:", image_data_right)
            Drone_test.show_image("right")

        # 深度摄像头测试
        waiting = True
        Running = True
        print(
                "Next we will test the depth camera\n"
                "You can always press p to continue and q to the next test in the pygame window\n"
                "Get depth camera function is class_name.get_raw_depth_image()\n"
                "(Press p to continue):")

        # 一直到按下p否则不动
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        waiting = False
                        break
                elif event.type == pygame.QUIT:
                    raise EndException

        while Running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        Running = False
                        break
                elif event.type == pygame.QUIT:
                    raise EndException

            image_data_depth = Drone_test.get_raw_depth_image()
            print("image_data_right:", image_data_depth)
            Drone_test.show_image("depth")

        print("=================================\n"
              "====== Camera Test is done ======\n"
              "=================================\n")
        """
        # 测试移动
        print("Next we will test the Drone Move.\n"
              "Please keep your Drone in the air.\n"
              "The Drone will move forward 0.5m.\n"
              "Press P when Drone is prepared.")

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        waiting = False
                        break
                elif event.type == pygame.QUIT:
                    raise EndException

        MoveSize = [0.5, 0, 0]
        Drone_test.Move(MoveSize, None)
        """

        print("The test process is end.\n"
              "Thank you for using test function.")
    except EndException:
        print("The test process is closed force")





'''
pygame.init()
screen = pygame.display.set_mode((640, 480))
Drone_test = Drone(True)
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    Drone_test.show_image("YOLO")
'''
