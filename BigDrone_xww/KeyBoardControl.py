from Ros_Drone_final_one import Drone

drone = Drone(False)
# drone.TakeOff()

while True:
    a = float(input("请输入要旋转的值："))
    if a != 0:
        drone.MoveInt(orientation=a)
    else:
        break
