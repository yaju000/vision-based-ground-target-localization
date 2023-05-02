#!/usr/bin/env python
# ground target tracking
import math
import numpy as np
from pandas import DataFrame
import rospy

from mavros_msgs.msg import State, AttitudeTarget, Thrust
from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.srv import ParamSet
from std_msgs.msg import Float64
from mavros_msgs.msg import Waypoint
from visualization_msgs.msg import Marker
from math import *
import datetime
import matplotlib.pyplot as plt
import time
import heapq
import tf

from AOA_v2 import AOA
from hsv_fw import hsv
from LeastQ_v1 import least_square
from dop_fw import path_planning

AOA = AOA()
hsv = hsv()
LQ = least_square()
cal_dop = path_planning()

enu_pos = []
azimuth = []
elevation = []
ob_points = []
uav_pose = []
rolllimit = []
vector = []
est_list = []
co = []
hd = []
vtp = []

starttime = datetime.datetime.now()

class aoa_info(object):
    def __init__(self):
        self.imu_msg = Imu()
        self.gps_msg = Odometry()
        self.hdg_msg = Float64()
        self.vel_msg = TwistStamped()

        self.gps_pose = [0,0,0]
        self.ned_pose = [0,0,0]
        self.vel_pose = [0,0,0]
        self.imu_x = 0
        self.lamda = 0
        self.index = 0
        self.roll, self.pitch, self.yaw = 0,0,0
        self.roll_n, self.pitch_e, self.yaw_d = 0, 0, 0
        self.roll_rate, self.pitch_rate, self.yaw_rate = 0, 0, 0
        self.quat = [0,0,0,0]
        self.u, self.v = 0, 0
        self.u_u, self.v_v = 0, 0
        self.P_img_x, self.P_img_y, self.P_img_z = 0, 0, 0
        self.angle_a_w, self.angle_e_w = 0, 0
        self.angle_a = [0, 0]
        self.angle_e = [0, 0]
        self.est_position = [0, 0, 0]
        self.est_x, self.est_y, self.est_z = 0, 0, 0
        self.est_n, self.est_e, self.est_d = 0, 0, 0
        self.est_vector_n, self.est_vector_e, self.est_vector_d = 0, 0, 0
        self.heading = 0
        self.vel = 0
        self.hd_deg = 0
        self.pro_vector = [0, 0]
        self.last_req = rospy.Time.now()

        ## Ros Service ##
        rospy.wait_for_service("/plane_cam_0/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("plane_cam_0/mavros/set_mode", SetMode)
        ## Subscribe Topic ##
        rospy.Subscriber("/plane_cam_0/mavros/state", State, self.state_cb)
        rospy.Subscriber("/plane_cam_0/mavros/imu/data", Imu, self.imu_callback)
        rospy.Subscriber("/plane_cam_0/mavros/global_position/local", Odometry, self.gps_callback)    
        rospy.Subscriber("/plane_cam_0/mavros/global_position/compass_hdg", Float64, self.hdg_callback)
        rospy.Subscriber("/plane_cam_0/mavros/local_position/velocity_local", TwistStamped, self.vel_callback)
        
        ## Publish Topic ##
        self.attitude_pub = rospy.Publisher("/plane_cam_0/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1)
        self.attitude = AttitudeTarget()
        self.attitude.type_mask = 0b0000011111 
        ## Record ##
        self.path_pub = rospy.Publisher('/plane_cam_0/mavros/setpoint_trajectory/desired', Path, queue_size=50)
        self.pose = PoseStamped()
        self.path_record = Path()
        self.marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=360)
        
    def state_cb(self, msg):
        self.current_state = msg

    def gps_callback(self, msg): #enu
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
        self.roll_rate = msg.angular_velocity.x
        self.pitch_rate = msg.angular_velocity.y
        self.yaw_rate = msg.angular_velocity.z

        self.roll, self.pitch, self.yaw = self.euler_from_quaternion(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        self.roll_n, self.pitch_e, self.yaw_d = self.ENU_to_NED(self.roll, self.pitch, self.yaw)

    def hdg_callback(self, msg): #enu # pi~(-pi)
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

    def vel_callback(self, msg): # enu
        self.vel_msg = msg
        self.vel_pose[0] = msg.twist.linear.x
        self.vel_pose[1] = msg.twist.linear.y
        self.vel_pose[2] = msg.twist.linear.z
        self.vel = np.sqrt(self.vel_pose[0]**2+self.vel_pose[1]**2)
        # print('vel =')
        # print(self.vel)

    def ENU_to_NED(self, x, y, z):
  
        R = [[0, 1, 0],[1, 0, 0],[0, 0, -1]]
        q = [x, y, z]
        ned = np.matmul(R,q)
        a = ned[0]
        b = ned[1]
        c = ned[2]
      
        return a, b, c

    def cal_aoa_info(self):

        if [self.u_u, self.v_v] != [0, 0]:
            ## Position_vector ##
            size_u = 320
            size_v = 240
            u_0 = size_u/2
            v_0 = size_v/2
            # focal length
            f = 277.191356
            self.P_img_x = u_0 - self.u_u
            self.P_img_y = v_0 - self.v_v 
            self.P_img_z = f
            P_img = [self.P_img_x, self.P_img_y, self.P_img_z] 
            # print("u, v = ")
            # print(self.u, self.v)
            # print('------------')
            # print("P_img = ")
            # print(P_img)

            P_img_1 = v_0 - 0
            P_img_2 = u_0 -160
            P_img_3 = f
            P_img_0 = [P_img_1, P_img_2, P_img_3]
            angle_ew_0 = AOA.AOA_v3(self.ned_pose[0], self.ned_pose[1], self.ned_pose[2], self.roll, self.pitch, self.yaw, P_img_1, P_img_2, P_img_3)
            self.angle_a_w, self.angle_e_w, self.angle_a, self.angle_e, self.est_n, self.est_e, self.est_d, self.est_vector_n, self.est_vector_e, self.est_vector_d = AOA.AOA_v0(self.ned_pose[0], self.ned_pose[1], self.ned_pose[2], self.roll, self.pitch, self.yaw, self.P_img_x, self.P_img_y, self.P_img_z)
            angle_a_w = self.angle_a_w
            angle_e_w = self.angle_e_w
            est_ned = [self.est_n, self.est_e, self.est_d]
            est_vector = [self.est_vector_n, self.est_vector_e, self.est_vector_d]
            ob_point = [self.ned_pose[0], self.ned_pose[1], self.ned_pose[2]]
            roll_limit = angle_ew_0 - angle_e_w
        
            ### R_ref ###
            R_ref = 50
            print('velocity =')
            print(self.vel)
            acc = self.vel*self.vel/R_ref
            print('------------------------')
            print('acc = ', acc)
            print('acc angle =')
            print(self.hd_deg+90)
            cx_n = acc*np.cos(self.hd_deg+90)
            cx_e = acc*np.sin(self.hd_deg+90)
            cx = np.sqrt(np.square(cx_n)+np.square(cx_e))
            c_n = np.abs(cx_n/cx)
            c_e = np.abs(cx_e/cx)
            # print('c_n = ', c_n)
            # print('c_e = ', c_e)
            if 0<=self.hd_deg<90:
                self.pro_vector = [-c_n, c_e]
            elif 90<=self.hd_deg<180:
                self.pro_vector = [-c_n, -c_e]
            elif 180<=self.hd_deg<270:
                self.pro_vector = [c_n, -c_e]  
            elif 270<=self.hd_deg<=360:
                self.pro_vector = [c_n, c_e]

            return angle_a_w, angle_e_w, est_ned, est_vector, ob_point, roll_limit

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

    def pathplanning(self, est, d, r, hd): #est_list[-1], distance, rolllimit, hd
        
        ############# waypoint #################
        value = []
        value_d = []
        next_position_list =  cal_dop.path_a(est_list[-1], d, r, hd)
        # print("next_position_list : ")
        # print(next_position_list)

        for i in range (3):
            a = [next_position_list[i*3][0], next_position_list[i*3][1], next_position_list[i*3][2]]
            b = [next_position_list[i*3+1][0], next_position_list[i*3+1][1], next_position_list[i*3+1][2]]
            c = [next_position_list[i*3+2][0], next_position_list[i*3+2][1], next_position_list[i*3+2][2]]

            GDOP, FIM = cal_dop.calculate_dop(a, b, c, est)
            value.append(GDOP)
            value_d.append(FIM)
        
        min_value = min(value)
        index = value.index(min_value)
        # print('index = ')
        # print(index)
        # print('min_value = ')
        # print(min_value)

        max_value = min(value_d)
        index_d = value_d.index(max_value)
        # print('value_d = ')
        # print(value_d)
        # rint('max_value = ')
        # print(max_value)

        next_1 = [next_position_list[index*3][0], next_position_list[index*3][1], next_position_list[index*3][2]]
        next_2 = [next_position_list[index*3+1][0], next_position_list[index*3+1][1], next_position_list[index*3+1][2]]
        next_3 = [next_position_list[index*3+2][0], next_position_list[index*3+2][1], next_position_list[index*3+2][2]]

        if index == 0:
            r_optimal = 50
        elif index == 1:
            r_optimal = 60
        elif index == 2:
            r_optimal = 70

        return r_optimal

    def height_control(self, pos, vel, K):

        h_error = 50-pos[2]
        pitch_d = (h_error/vel)*K
   
        return pitch_d

    def cal_vtp(self, distance, p, est, hd_enu):
        L1 = 70
        Radius = 100

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
        vtp.append([vtp_e, vtp_n])

        return vtp_e, vtp_n

    def Attitude_control(self,distance,vtp_e, vtp_n, hd_enu, vel, p):
        L1 = 70
        Radius = 100
        
        circle_error = distance - Radius
        desired_yaw = np.arctan2(vtp_n-p[1], vtp_e-p[0])  # ENU
        gamma = hd_enu - desired_yaw

        ####### Accerelation term #######
        u = 2*np.square(vel)*np.sin(gamma)/L1 + 0.05*circle_error

        roll_cmd = np.arctan(u/9.81)
        pitch_cmd = self.height_control(p, vel, 0.01)
        pitch_cmd = pitch_cmd*np.pi/180

        if roll_cmd > 0.785:
            roll_cmd = 0.785
        elif roll_cmd < -0.785:
            roll_cmd = -0.785
        else:
            roll_cmd = roll_cmd

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

    def record(self, yaw, est_list, Radius):

        current_time = rospy.Time.now()
        br = tf.TransformBroadcaster()
        # translate matrix
        br.sendTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                        rospy.Time.now(), "odom", "map")

        quat = tf.transformations.quaternion_from_euler(0, 0, yaw)

        for i in range(5):
            path_e = est_list[-1][1] + Radius*np.sin((yaw-np.pi/2)+72*i*np.pi/180)
            path_n = est_list[-1][0] + Radius*np.cos((yaw-np.pi/2)+72*i*np.pi/180)

            self.pose.header.stamp = current_time
            self.pose.header.frame_id = 'odom'
            self.pose.pose.position.x = path_e
            self.pose.pose.position.y = path_n
            self.pose.pose.position.z = 50
            self.pose.pose.orientation.x = quat[0]
            self.pose.pose.orientation.y = quat[1]
            self.pose.pose.orientation.z = quat[2]
            self.pose.pose.orientation.w = quat[3]
            # path setting
            self.path_record.header.frame_id = 'odom'
            self.path_record.header.stamp = current_time
            self.path_record.poses.append(self.pose)
            # number of path 
            if len(self.path_record.poses) > 10:
                self.path_record.poses.pop(0)
            self.path_pub.publish(self.path_record)

    def iteration(self, event):

        aa = self.hd_deg
        # print('heading angle =')
        # print(aa)

        self.u, self.v = hsv.value_callback()

        if [self.u, self.v]!=[self.u_u, self.v_v]:
            self.u_u = self.u
            self.v_v = self.v

            ### aoa ###
            est_vector = [self.est_vector_n, self.est_vector_e]
            pro_vector = self.pro_vector
            # print('------------------------')
            # print('est_vector = ')
            # print(est_vector)
            # print('pro_vector = ')
            # print(pro_vector)
            print('------------------------')
            print('Number of observation points =')
            print(len(azimuth))
            print('Number of iteration =')
            print(len(est_list))
            print('------------------------')

            if  10 >= len(azimuth)>=2:
  
                if self.hd_deg % 60 < 2:
                    angle_a_w, angle_e_w, est_ned, est_vector, ob_point, roll_limit = self.cal_aoa_info()
                    #print('roll_limit =', roll_limit)
                    azimuth.append(angle_a_w)
                    ob_points.append(ob_point)
                    uav_pose.append([self.roll, self.pitch, self.yaw])
                    vector.append(est_vector)

                if self.hd_deg % 360 < 2:
                    Est_n,Est_e,Est_d = LQ.LeastQ_m(ob_points, azimuth)
                    Est_position = [Est_n,Est_e,Est_d]
                    est_list.append(Est_position)
                    print('------------------------')
                    print('est_list =', est_list)
                    print('------------------------')

                n,e = est_list[-1][0],est_list[-1][1]
                est = [e, n, 0]

                ##### start path planning #####
                print('=== Image successful!===')
                print('=== Go to loiter the estimated target ===')
                print('est_target position =', est)
                print('------------------------')
                ## distance between uav and estimated target
                distance = np.sqrt(np.square(self.gps_pose[1]-n)+np.square(self.gps_pose[0]-e))
                # print('distance = ')
                # print(distance)
                p = self.gps_pose

                ### Find the reference point ###
                vtp_e, vtp_n = self.cal_vtp(distance, self.gps_pose, est, self.heading)
               
                ### Attitude control ###
                circle_error, u, roll_cmd, pitch_cmd = self.Attitude_control(distance, vtp_e, vtp_n, self.heading, self.vel, p)
             
                if (self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - self.last_req) > rospy.Duration(0.5)):
                    offb_set_mode = SetModeRequest()
                    offb_set_mode.custom_mode = 'OFFBOARD'
                    if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
                        rospy.loginfo("OFFBOARD enabled")
            
                    self.last_req = rospy.Time.now()
                else:
                    quat = tf.transformations.quaternion_from_euler(roll_cmd, pitch_cmd, 0) # roll pitch yaw
                    self.attitude.orientation.x = quat[0]
                    self.attitude.orientation.y = quat[1]
                    self.attitude.orientation.z = quat[2]
                    self.attitude.orientation.w = quat[3]
                    self.attitude.thrust = 0.35
                    self.attitude_pub.publish(self.attitude)

                    ############# Record ###############
                    self.record(self.heading, est_list, 100)
                    ####################################

                    # endtime = datetime.datetime.now()
                    # print ('Total time =')
                    # print (endtime - starttime).seconds

            elif  len(azimuth) < 2: # No.3 observation
                
                if len(azimuth) == 0:
                    if [self.u, self.v] != [0, 0]:
                        angle_a_w, angle_e_w, est_ned, est_vector, ob_point, roll_limit = self.cal_aoa_info()
                        azimuth.append(angle_a_w)
                        elevation.append(angle_e_w)
                        ob_points.append(ob_point)
                        vector.append(est_vector)
                        est_list.append(est_ned)
                        rolllimit.append(roll_limit)
                        uav_pose.append([self.roll, self.pitch, self.yaw])
                        
                        if len(hd) == 0:
                            hd.append(self.hd_deg)

                        print('azimuth =')
                        print(azimuth)
                    else:
                        pass
    
                elif len(azimuth) == 1:
                    print('hd change =')
                    print(np.abs(hd[0] - self.hd_deg))
                    if np.abs(hd[0] - self.hd_deg) >= 90:
                        angle_a_w, angle_e_w, est_ned, est_vector, ob_point, roll_limit = self.cal_aoa_info()
                        azimuth.append(angle_a_w)
                        elevation.append(angle_e_w)
                        ob_points.append(ob_point)
                        vector.append(est_vector)
                        rolllimit.append(roll_limit)
                        uav_pose.append([self.roll, self.pitch, self.yaw])
                        Est_n,Est_e,Est_d = LQ.LeastQ_m(ob_points, azimuth)
                        Est_position = [Est_n,Est_e,Est_d]
                        est_list.append(Est_position)

                    ### Go to No.2 observation point ###
                    k = 1.0
                    sin_gamma = (pro_vector[0]*est_vector[1]-pro_vector[1]*est_vector[0])/(np.sqrt(np.square(pro_vector[0])+np.square(pro_vector[1]))*np.sqrt(np.square(est_vector[0])+np.square(est_vector[1])))
                    acc_cmd = np.square(self.vel)/50 + k*sin_gamma
                    roll_cmd = np.arctan(acc_cmd/9.81)

                    pitch_cmd = self.height_control(self.gps_pose, self.vel, 0.01)

                    if roll_cmd > rolllimit[0]:
                        roll_cmd = rolllimit[0]
                    elif roll_cmd < -rolllimit[0]:
                        roll_cmd = -rolllimit[0]

                    print('------visual servo-----------')
                    print('sin_gamma =', sin_gamma)
                    print('acc_cmd =', acc_cmd)
                    print('roll_cmd =', roll_cmd)
                    print('pitch_cmd =', pitch_cmd)
                    print('-----------------------------')

                    if (self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - self.last_req) > rospy.Duration(0.5)):
                        offb_set_mode = SetModeRequest()
                        offb_set_mode.custom_mode = 'OFFBOARD'
                        if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
                            rospy.loginfo("OFFBOARD enabled")
                        self.last_req = rospy.Time.now()
                    else:
                        quat = tf.transformations.quaternion_from_euler(roll_cmd, pitch_cmd, 0) # roll pitch yaw
                        self.attitude.orientation.x = quat[0]
                        self.attitude.orientation.y = quat[1]
                        self.attitude.orientation.z = quat[2]
                        self.attitude.orientation.w = quat[3]
                        self.attitude.thrust = 0.35
                        self.attitude_pub.publish(self.attitude)
            else:
                print('=== Mission finished ===')
                                 
                n,e = est_list[-1][0],est_list[-1][1]
                est = [e, n, 0]
                print('------------------------')
                print('final estimated target position = ', est)
                print('------------------------')
               
                ## distance between uav and estimated target
                distance = np.sqrt(np.square(self.gps_pose[1]-n)+np.square(self.gps_pose[0]-e))
                p = self.gps_pose

                ### Find the reference point ###
                vtp_e, vtp_n = self.cal_vtp(distance, self.gps_pose, est, self.heading)
               
                ### Attitude control ###
                circle_error, u, roll_cmd, pitch_cmd = self.Attitude_control(distance, vtp_e, vtp_n, self.heading, self.vel, p)

                if (self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - self.last_req) > rospy.Duration(0.5)):
                    offb_set_mode = SetModeRequest()
                    offb_set_mode.custom_mode = 'OFFBOARD'
                    if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
                        rospy.loginfo("OFFBOARD enabled")
            
                    self.last_req = rospy.Time.now()
                else:
                    quat = tf.transformations.quaternion_from_euler(roll_cmd, pitch_cmd, 0) # roll pitch yaw
                    self.attitude.orientation.x = quat[0]
                    self.attitude.orientation.y = quat[1]
                    self.attitude.orientation.z = quat[2]
                    self.attitude.orientation.w = quat[3]
                    self.attitude.thrust = 0.35
                    self.attitude_pub.publish(self.attitude)

        else:
            if len(est_list) == 0:
                print('=== No target!! ===')
            
            elif len(est_list) < 2:
                print('=== Target dispears!! ===')
                print('=== keep target in camera view! ===')
                est_vector = [self.est_vector_n, self.est_vector_e]
                pro_vector = self.pro_vector

                k = 1.0
                sin_gamma = (pro_vector[0]*est_vector[1]-pro_vector[1]*est_vector[0])/(np.sqrt(np.square(pro_vector[0])+np.square(pro_vector[1]))*np.sqrt(np.square(est_vector[0])+np.square(est_vector[1])))
                acc_cmd = np.square(self.vel)/50 + k*sin_gamma
                roll_cmd = np.arctan(acc_cmd/9.81)

                pitch_cmd = self.height_control(self.gps_pose, self.vel, 0.01)

                if roll_cmd > rolllimit[0]:
                    roll_cmd = rolllimit[0]
                elif roll_cmd < -rolllimit[0]:
                    roll_cmd = -rolllimit[0]

                print('------visual servo-----------')
                print('sin_gamma =', sin_gamma)
                print('acc_cmd =', acc_cmd)
                print('roll_cmd =', roll_cmd)
                print('pitch_cmd =', pitch_cmd)
                print('-----------------------------')

                if (self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - self.last_req) > rospy.Duration(0.5)):
                    offb_set_mode = SetModeRequest()
                    offb_set_mode.custom_mode = 'OFFBOARD'
                    if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
                        rospy.loginfo("OFFBOARD enabled")
                    self.last_req = rospy.Time.now()
                else:
                    quat = tf.transformations.quaternion_from_euler(roll_cmd, pitch_cmd, 0) # roll pitch yaw
                    self.attitude.orientation.x = quat[0]
                    self.attitude.orientation.y = quat[1]
                    self.attitude.orientation.z = quat[2]
                    self.attitude.orientation.w = quat[3]
                    self.attitude.thrust = 0.35
                    self.attitude_pub.publish(self.attitude)
                
            elif len(est_list) >= 2 and len(azimuth) <= 10:

                print('=== Target dispears!! ===')
                print('=== Go to loiter the estimated target ===')

                ## Go back to last eastimated target position ## 
                n,e = est_list[-1][0],est_list[-1][1]
                est = [e, n, 0]
                print('est_target position =', est)
                print('------------------------')

                ## distance between uav and estimated target
                distance = np.sqrt(np.square(self.gps_pose[1]-n)+np.square(self.gps_pose[0]-e))
                p = self.gps_pose

                ### Find the reference point ###
                vtp_e, vtp_n = self.cal_vtp(distance, self.gps_pose, est, self.heading)
               
                ### Attitude control ###
                circle_error, u, roll_cmd, pitch_cmd = self.Attitude_control(distance, vtp_e, vtp_n, self.heading, self.vel, p)

                if (self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - self.last_req) > rospy.Duration(0.5)):
                    offb_set_mode = SetModeRequest()
                    offb_set_mode.custom_mode = 'OFFBOARD'
                    if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
                        rospy.loginfo("OFFBOARD enabled")
            
                    self.last_req = rospy.Time.now()
                else:
                    quat = tf.transformations.quaternion_from_euler(roll_cmd, pitch_cmd, 0) # roll pitch yaw
                    self.attitude.orientation.x = quat[0]
                    self.attitude.orientation.y = quat[1]
                    self.attitude.orientation.z = quat[2]
                    self.attitude.orientation.w = quat[3]
                    self.attitude.thrust = 0.35
                    self.attitude_pub.publish(self.attitude)

                    ############# Record ###############
                    self.record(self.heading, est_list, 100)
                    ####################################

if __name__ == '__main__':

    rospy.init_node('main_system', anonymous=True)
    dt = 1.0/20
    pathplan_run = aoa_info()
    rospy.Timer(rospy.Duration(dt), pathplan_run.iteration)
    rospy.spin()

    df = DataFrame({'enu_pos': enu_pos})
    df.to_excel('fw_tt_path.xlsx', sheet_name='sheet1', index=False)
    dq = DataFrame({'azimuth':azimuth,'ob_points': ob_points,'uav_pose':uav_pose})
    dq.to_excel('fw_tt_result.xlsx', sheet_name='sheet1', index=False)
    dm = DataFrame({'vtp':vtp})
    dm.to_excel('fw_tt_vtp.xlsx', sheet_name='sheet1', index=False)
    dr = DataFrame({'est_list':est_list})
    dr.to_excel('fw_tt_est.xlsx', sheet_name='sheet1', index=False)



