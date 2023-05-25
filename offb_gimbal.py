#!/usr/bin/env python
# ground target tracking
import math
import numpy as np
import rospy
from pandas import DataFrame
from mavros_msgs.msg import State, AttitudeTarget, Thrust, ActuatorControl
from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped, PointStamped, Quaternion
from nav_msgs.msg import Odometry, Path
from mavros_msgs.msg import PositionTarget, MountControl
from mavros_msgs.srv import SetMode, SetModeRequest, SetMavFrame
from mavros_msgs.srv import ParamSet
from std_msgs.msg import Float64
from mavros_msgs.msg import Waypoint
from math import *
from datetime import datetime
import time
import heapq
import tf

enu_pos = []
yaw_list = []

class aoa_info(object):

    def __init__(self):
        self.imu_msg = Imu()
        self.gps_msg = Odometry()
        self.hdg_msg = Float64()
        self.vel_msg = TwistStamped()

        self.gps_pose = [0,0,0]
        self.ned_pose = [0,0,0]
        self.vel_pose = [0,0,0]
        self.quat = [0,0,0,0]
        self.tracking_pos = [0,0,0]

        self.heading = 0
        self.vel = 0
        self.last_req = rospy.Time.now()

        rospy.Subscriber("/plane_cam_0/mavros/state", State, self.state_cb)
        rospy.Subscriber("/plane_cam_0/mavros/imu/data", Imu, self.imu_callback)
        rospy.Subscriber("/plane_cam_0/mavros/global_position/local", Odometry, self.gps_callback)    
        rospy.Subscriber("/plane_cam_0/mavros/global_position/compass_hdg", Float64, self.hdg_callback)
        rospy.Subscriber("/plane_cam_0/mavros/local_position/velocity_local", TwistStamped, self.vel_callback)
        rospy.Subscriber("point_of_interest", PointStamped, self.point_callback)
        #rospy.Subscriber("/plane_cam_0/mavros/mount_control/orientation", Quaternion, self.gb_callback)
        
        rospy.wait_for_service("/plane_cam_0/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("plane_cam_0/mavros/set_mode", SetMode)

        self.attitude_pub = rospy.Publisher("/plane_cam_0/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1)
        self.attitude = AttitudeTarget()
        self.attitude.type_mask = 0b0000011111 
        ### Gimbal ###
        self.gimbal_pub = rospy.Publisher("/plane_cam_0/mavros/mount_control/command", MountControl, queue_size=1)
        self.gimbal = MountControl()
        self.gimbal.header.stamp = rospy.Time.now()
        self.gimbal.header.frame_id = 'map'
        self.gimbal.mode = 2
        self.Actuator_pub = rospy.Publisher("/plane_cam_0/mavros/actuator_control", ActuatorControl, queue_size=1)
        self.Actuator = ActuatorControl()
        self.Actuator.header.stamp = rospy.Time.now()
        self.Actuator.header.frame_id = 'map'
        self.Actuator.group_mix = 2
        
    # def gb_callback(self, msg):


    def point_callback(self, msg):
        self.tracking_pos[0] = msg.point.x
        self.tracking_pos[1] = msg.point.y
        self.tracking_pos[2] = msg.point.z

    def state_cb(self, msg):
        self.current_state = msg

    def gps_callback(self, msg):
        self.gps_msg = msg
        self.gps_pose[0] = msg.pose.pose.position.x
        self.gps_pose[1] = msg.pose.pose.position.y
        self.gps_pose[2] = msg.pose.pose.position.z

        self.ned_pose[0], self.ned_pose[1], self.ned_pose[2] = self.gps_pose[1], self.gps_pose[0], -self.gps_pose[2]
        enu = [self.gps_pose[0], self.gps_pose[1], self.gps_pose[2]]
        enu_pos.append(enu)

    def imu_callback(self, msg):
        self.imu_msg = msg
        self.quat[0] = msg.orientation.w
        self.quat[1] = msg.orientation.x
        self.quat[2] = msg.orientation.y
        self.quat[3] = msg.orientation.z

        self.roll, self.pitch, self.yaw = self.euler_from_quaternion(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)

    def hdg_callback(self, msg): #enu
        self.hdg_msg = msg
        heading_angle = msg.data #ned
        self.hd_deg = heading_angle #ned # degree
        self.heading = 90-heading_angle 

        if self.heading <= -180:
            self.heading = self.heading + 360
        else:
            self.heading = self.heading

        self.heading = np.deg2rad(self.heading) #enu

        # print('heading angle(deg) = ')
        # print(heading_angle)
        # print('heading angle(rad) = ')
        # print(self.heading)

    def vel_callback(self, msg): #ned
        self.vel_msg = msg
        self.vel_pose[0] = msg.twist.linear.x
        self.vel_pose[1] = msg.twist.linear.y
        self.vel_pose[2] = msg.twist.linear.z
        self.vel = np.sqrt(self.vel_pose[0]**2+self.vel_pose[1]**2)
        #print('self.vel_pose')
        #print(self.vel_pose)

    def ENU_to_NED(self, x, y, z):
  
        R = [[0, 1, 0],[1, 0, 0],[0, 0, -1]]
        q = [x, y, z]
        ned = np.matmul(R,q)
        a = ned[0]
        b = ned[1]
        c = ned[2]
      
        return a, b, c

    def NED_to_ENU(self, x, y, z):
  
        R = [[0, 1, 0],[1, 0, 0],[0, 0, -1]]
        q = [x, y, z]
        ned = np.matmul(R,q)
        a = ned[0]
        b = ned[1]
        c = ned[2]
      
        return a, b, c

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

    def vision_to_world(self, roll, pitch, yaw):
  
        R_x = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
        R_y = [[cos(10), 0, -sin(10)],[0, 1, 0],[sin(10), 0, cos(10)]]
        R_z = [[cos(90), -sin(90), 0],[sin(90), cos(90), 0],[0, 0, 1]]
        R = np.matmul(R_y, R_z)
        #print(R)
        
        A = [roll, pitch, yaw]
        [vision_roll, vision_pitch, vision_yaw] = np.matmul(R,A)

        return vision_roll, vision_pitch, vision_yaw

    def insec(self,p1, r1, p2, r2):
            x = p1[0]
            y = p1[1]
            R = r1
            a = p2[0]
            b = p2[1]
            S = r2
            d = np.sqrt((np.abs(a - x)) ** 2 + (np.abs(b - y)) ** 2)
            if d > (R + S) or d < (np.abs(R - S)):
                #print("Two circles have no intersection")
                return None,None
            elif d == 0:
                #print("Two circles have same center!")
                return None,None
            else:
                A = (R ** 2 - S ** 2 + d ** 2) / (2 * d)
                h = np.sqrt(R ** 2 - A ** 2)
                x2 = x + A * (a - x) / d
                y2 = y + A * (b - y) / d
                x3 = round(x2 - h * (b - y) / d, 2)
                y3 = round(y2 + h * (a - x) / d, 2)
                x4 = round(x2 + h * (b - y) / d, 2)
                y4 = round(y2 - h * (a - x) / d, 2)
                c1 = [x3, y3]
                c2 = [x4, y4]
            return c1, c2

    def height_control(self, pos, vel, K):

        h_error = 50-pos[2]
        pitch_d = (h_error/vel)*K
   
        return pitch_d

    def cal_vtp(self, distance, p, Radius, est, hd_enu):

        L1 = 50
        vtp_1, vtp_2 = self.insec(p, L1, est, Radius)

        if [vtp_1, vtp_2] == [None, None]:
            vtp_1, vtp_2 = self.insec(p, distance, est, Radius)
        else:
            pass
        #####################################################
        hd_point = [p[0]+np.cos(hd_enu), p[1]+sin(hd_enu)]
        #print('hd_point =', hd_point)
        L_0 = [hd_point[0]-p[0], hd_point[1]-p[1]]
        L_1 = [vtp_1[0]-p[0], vtp_1[1]-p[1]]
        L_2 = [vtp_2[0]-p[0], vtp_2[1]-p[1]]

        cos_1 = np.dot(L_0, L_1)
        cos_2 = np.dot(L_0, L_2)
        # print('----------------------')
        # print('cos_1 =', cos_1)
        # print('cos_2 =', cos_2)
        # print('----------------------')

        if cos_1 > cos_2 :
            vtp_n = vtp_1[1]
            vtp_e = vtp_1[0]

        elif cos_1 < cos_2 :
            vtp_n = vtp_2[1]
            vtp_e = vtp_2[0]
        #####################################################
        print('----------------------')
        print('vtp_e, vtp_n = ')
        print(vtp_e, vtp_n)
        print('----------------------')

        return vtp_e, vtp_n

    def Attitude_control(self, distance, Radius,vtp_e, vtp_n, hd_enu, vel, p):

        L1 = 50
        circle_error = distance - Radius
        desired_yaw = np.arctan2(vtp_n-p[1], vtp_e-p[0])  # ENU
        gamma = hd_enu - desired_yaw

        ####### Accerelation term #######
        u = 2*np.square(vel)*np.sin(gamma)/L1 + 0.05*circle_error

        roll_cmd = np.arctan(u/9.81)
        pitch_cmd = self.height_control(p, vel, 0.05)
        pitch_cmd = pitch_cmd*np.pi/180

        ###### new #######
        if roll_cmd > 0.785:
            roll_cmd = 0.785
        elif roll_cmd < -0.785:
            roll_cmd = -0.785

        if pitch_cmd > 0.174:
            pitch_cmd = 0.174
        elif pitch_cmd < -0.174:
            pitch_cmd = -0.174
        else:
            pitch_cmd = pitch_cmd

        print('circle_error =', circle_error)
        print('u =',u)
        print('roll_cmd =', roll_cmd)
        print('pitch_cmd =', pitch_cmd)

        return circle_error, u, roll_cmd, pitch_cmd

    def iteration(self, event):
        print('roll, pitch, yaw = ')
        print(self.roll, self.pitch, self.yaw)
        
        #inputs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        est = [50, 0,0]
        p = self.gps_pose
        g = 9.81
        R = 70
        distance = np.sqrt(np.square(p[1]-est[1])+np.square(p[0]-est[0]))
        # print('distance = ',distance)

        ### Find the reference point ###
        vtp_e, vtp_n = self.cal_vtp(distance, self.gps_pose, R, est, self.heading)
        ### Attitude control ###
        circle_error, u, roll_cmd, pitch_cmd = self.Attitude_control(distance, R, vtp_e, vtp_n, self.heading, self.vel, p)
        #### Gimbal ######
        if self.roll >= 0:
            g_yaw  = 90
        else:
            g_yaw  = -90
        ##################
        if (self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - self.last_req) > rospy.Duration(0.5)):
            offb_set_mode = SetModeRequest()
            offb_set_mode.custom_mode = 'OFFBOARD'
            if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
                rospy.loginfo("OFFBOARD enabled")

            self.last_req = rospy.Time.now()
        else:
            # quat = tf.transformations.quaternion_from_euler(roll_cmd, pitch_cmd, 0) # roll pitch yaw
            # self.attitude.orientation.x = quat[0]
            # self.attitude.orientation.y = quat[1]
            # self.attitude.orientation.z = quat[2]
            # self.attitude.orientation.w = quat[3]
            # self.attitude.thrust = 0.4
            # self.attitude_pub.publish(self.attitude)

            self.gimbal.roll = 0
            self.gimbal.pitch = 1.57/np.pi*180
            self.gimbal.yaw = 1.57/np.pi*180
            self.gimbal_pub.publish(self.gimbal)

            # self.Actuator.controls[1] = 1.57/np.pi
            # self.Actuator.controls[2] = 1.57/np.pi
            # self.Actuator_pub.publish(self.Actuator)


if __name__ == '__main__':
    
    rospy.init_node('offb_gimbal_py', anonymous=True)
    dt = 1.0/20
    pathplan_run = aoa_info()
    rospy.Timer(rospy.Duration(dt), pathplan_run.iteration)
    rospy.spin()

