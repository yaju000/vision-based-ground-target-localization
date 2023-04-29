#!/usr/bin/env python
# ground target tracking
import math
import numpy as np
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
yaw_rate_list = []
sin_list = []
plan_1 = []
Q_list = []
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
        self.vision_roll, self.vision_pitch, self.vision_yaw = 0, 0, 0
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
        rospy.Subscriber("/plane_cam_0/mavros/global_position/raw/gps_vel", TwistStamped, self.vel_callback)
        #rospy.Subscriber("/plane_cam_0/mavros/setpoint_raw/local", PositionTarget, self.pose_callback)
        ## Publish Topic ##
        self.setpoint_pub = rospy.Publisher("/plane_cam_0/mavros/setpoint_raw/local", PositionTarget, queue_size=1)
        self.setpoint = PositionTarget()
        self.setpoint.coordinate_frame = 1
        #self.setpoint.coordinate_frame = self.coordinate_frame 
        # print('self.setpoint.coordinate_frame = ')
        # print(self.setpoint.coordinate_frame)
        self.attitude_pub = rospy.Publisher("/plane_cam_0/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1)
        self.attitude = AttitudeTarget() 
        ## Record ##
        self.path_pub = rospy.Publisher('/plane_cam_0/mavros/setpoint_trajectory/desired', Path, queue_size=50)
        self.pose = PoseStamped()
        self.path_record = Path()
        self.marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)
        
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
        # print(self.roll_rate,self.pitch_rate,self.yaw_rate)

    def hdg_callback(self, msg): #enu # pi~(-pi)
        self.hdg_msg = msg
        heading_angle = msg.data
        self.hd_deg = heading_angle
        if heading_angle > 180:
            self.heading = np.deg2rad(heading_angle-360)
        else:
            self.heading = np.deg2rad(heading_angle)

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

    def cal_aoa_info(self):

        if [self.u_u, self.v_v] != [0, 0]:
            ## Position_vector ##
            size_u = 320
            size_v = 240
            u_0 = size_u/2
            v_0 = size_v/2
            # focal length
            f = 277.191356
            self.P_img_x = v_0 - self.v_v
            self.P_img_y = self.u_u - u_0
            self.P_img_z = f
            P_img = [self.P_img_x, self.P_img_y, self.P_img_z] 
            # print("u, v = ")
            # print(self.u, self.v)
            # print('------------')
            # print("P_img = ")
            # print(P_img)

            P_img_1 = v_0 - 0
            P_img_2 = 160 - u_0
            P_img_3 = f
            P_img_0 = [P_img_1, P_img_2, P_img_3]
            angle_ew_0 = AOA.AOA_v3(self.ned_pose[0], self.ned_pose[1], self.ned_pose[2], self.roll, self.pitch, self.yaw, P_img_1, P_img_2, P_img_3)
            self.angle_a_w, self.angle_e_w, self.angle_a, self.angle_e, self.est_n, self.est_e, self.est_d, self.est_vector_n, self.est_vector_e, self.est_vector_d = AOA.AOA_v2(self.ned_pose[0], self.ned_pose[1], self.ned_pose[2], self.roll, self.pitch, self.yaw, self.P_img_x, self.P_img_y, self.P_img_z)
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

    def draw_curve(p1, p2):
        
        a = (p2[1] - p1[1])/ (np.cosh(p2[0]) - np.cosh(p1[0]))
        b = p1[1] - a * np.cosh(p1[0])
        x = np.linspace(p1[0], p2[0], 100)
        y = a * np.cosh(x) + b
        
        return x, y

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
            print('------------------------')

            if  10 >= len(azimuth)>=2:
  
                # if self.hd_deg % 60 == 0:
                #     angle_a_w, angle_e_w, est_ned, est_vector, ob_point, roll_limit = self.cal_aoa_info()
                #     print('roll_limit =', roll_limit)
                #     azimuth.append(angle_a_w)
                #     ob_points.append(ob_point)
                #     vector.append(est_vector)
                #     est_list.append(est_ned)
                
                if 119 < self.hd_deg < 121 :
                    angle_a_w, angle_e_w, est_ned, est_vector, ob_point, roll_limit = self.cal_aoa_info()
                    azimuth.append(angle_a_w)
                    ob_points.append(ob_point)
                    vector.append(est_vector)
                    est_list.append(est_ned)
                    uav_pose.append([self.roll, self.pitch, self.yaw])
                    Est_n,Est_e,Est_d = LQ.LeastQ_m(ob_points, azimuth)
                    Est_position = [Est_n,Est_e,Est_d]
                    est_list.append(Est_position)

                elif 59 < self.hd_deg < 61 :
                    angle_a_w, angle_e_w, est_ned, est_vector, ob_point, roll_limit = self.cal_aoa_info()
                    azimuth.append(angle_a_w)
                    ob_points.append(ob_point)
                    vector.append(est_vector)
                    est_list.append(est_ned)
                    uav_pose.append([self.roll, self.pitch, self.yaw])
                    Est_n,Est_e,Est_d = LQ.LeastQ_m(ob_points, azimuth)
                    Est_position = [Est_n,Est_e,Est_d]
                    est_list.append(Est_position)

                elif 0 < self.hd_deg < 1 :
                    angle_a_w, angle_e_w, est_ned, est_vector, ob_point, roll_limit = self.cal_aoa_info()
                    azimuth.append(angle_a_w)
                    ob_points.append(ob_point)
                    vector.append(est_vector)
                    est_list.append(est_ned)
                    uav_pose.append([self.roll, self.pitch, self.yaw])
                    Est_n,Est_e,Est_d = LQ.LeastQ_m(ob_points, azimuth)
                    Est_position = [Est_n,Est_e,Est_d]
                    est_list.append(Est_position)

                elif 179 < self.hd_deg < 181 :
                    angle_a_w, angle_e_w, est_ned, est_vector, ob_point, roll_limit = self.cal_aoa_info()
                    azimuth.append(angle_a_w)
                    ob_points.append(ob_point)
                    vector.append(est_vector)
                    est_list.append(est_ned)
                    uav_pose.append([self.roll, self.pitch, self.yaw])
                    Est_n,Est_e,Est_d = LQ.LeastQ_m(ob_points, azimuth)
                    Est_position = [Est_n,Est_e,Est_d]
                    est_list.append(Est_position)

                elif 239 < self.hd_deg < 241 :
                    angle_a_w, angle_e_w, est_ned, est_vector, ob_point, roll_limit = self.cal_aoa_info()
                    azimuth.append(angle_a_w)
                    ob_points.append(ob_point)
                    vector.append(est_vector)
                    est_list.append(est_ned)
                    uav_pose.append([self.roll, self.pitch, self.yaw])
                    Est_n,Est_e,Est_d = LQ.LeastQ_m(ob_points, azimuth)
                    Est_position = [Est_n,Est_e,Est_d]
                    est_list.append(Est_position)
                    
                elif 299 < self.hd_deg < 301 :
                    angle_a_w, angle_e_w, est_ned, est_vector, ob_point, roll_limit = self.cal_aoa_info()
                    azimuth.append(angle_a_w)
                    ob_points.append(ob_point)
                    vector.append(est_vector)
                    est_list.append(est_ned)
                    uav_pose.append([self.roll, self.pitch, self.yaw])
                    Est_n,Est_e,Est_d = LQ.LeastQ_m(ob_points, azimuth)
                    Est_position = [Est_n,Est_e,Est_d]
                    est_list.append(Est_position)

                n,e = est_list[-1][0],est_list[-1][1]
                print('------------------------')
                print('est_target position = ')
                print(n,e)
                est = [e, n, 0]

                ##### start path planning #####
                ## distance between uav and estimated target
                distance = np.sqrt(np.square(self.gps_pose[1]-n)+np.square(self.gps_pose[0]-e))
                # print('distance = ')
                # print(distance)

                ### step1 : find optimal circle radius ###
                #r_optimal = self.pathplanning(est_list[-1], distance, rolllimit, hd)
                # if self.roll < rolllimit : 
                if distance >= 50 :
                    r_optimal = distance
                else:
                    r_optimal = 50

                print('r_optimal =')
                print(r_optimal)
                ### step2 : Go to the optimal circle ###
                ## waypoint Control ##
                L = distance
                p = self.gps_pose
                vtp_1, vtp_2 = self.insec(p, L, est, L)
                
                uav_to_vtp_1 = np.arctan2(vtp_1[0]-p[0], vtp_1[1]-p[1]) 
                uav_to_vtp_2 = np.arctan2(vtp_2[0]-p[0], vtp_2[1]-p[1]) 
                # print('----------------------')
                # print('uav_to_vtp_1, uav_to_vtp_2 = ')
                # print(uav_to_vtp_1, uav_to_vtp_2)
                # print('self.heading = ')
                # print(self.heading)

                angle_1 = self.heading - uav_to_vtp_1
                angle_2 = self.heading - uav_to_vtp_2
                # print('angle_1, angle_2 = ')
                # print(angle_1, angle_2)

                if self.heading > 0:
                    if np.abs(angle_1) < np.abs(angle_2):
                        vtp_n = vtp_1[1]
                        vtp_e = vtp_1[0]
                    else:
                        vtp_n = vtp_2[1]
                        vtp_e = vtp_2[0]
                else:
                    if angle_1 > angle_2:
                        vtp_n = vtp_1[1]
                        vtp_e = vtp_1[0]
                    else:
                        vtp_n = vtp_2[1]
                        vtp_e = vtp_2[0]
                print('----------------------')
                print('vtp_e, vtp_n = ')
                print(vtp_e, vtp_n)
                print('----------------------')

                vtp.append([vtp_e, vtp_n])
                ############ Calculate FIM ##############
                # Q = cal_dop.cal_H(self.gps_pose, [50, 0, 0])
                # H.append(Q)
                # total_H = sum(H)
                # print('total_H =')
                # print(total_H)
                # co.append(total_H)
                # print('FIM =')
                # print(co[-1]*(1/np.square(error1)))
                # covariance = np.linalg.det(co[-1]*(1/np.square(error1)))
                # print('covariance =')
                # print(covariance)
                # FIM_det = -math.log(covariance)
                # print('FIM_det =')
                # print(FIM_det)
                #################################

                if (self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - self.last_req) > rospy.Duration(0.5)):
                    offb_set_mode = SetModeRequest()
                    offb_set_mode.custom_mode = 'OFFBOARD'
                    if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
                        rospy.loginfo("OFFBOARD enabled")
            
                    self.last_req = rospy.Time.now()
                else:
                    print('loiter to estimated point')
                    self.setpoint.position.x = vtp_e
                    self.setpoint.position.y = vtp_n
                    self.setpoint.position.z = 50
                    self.setpoint_pub.publish(self.setpoint)

                    ############# Record ####################
                    current_time = rospy.Time.now()
                    br = tf.TransformBroadcaster()
                    # translate matrix
                    br.sendTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                                    rospy.Time.now(), "odom", "map")

                    if 0 <= self.hd_deg <= 90:
                        yaw = (90- self.hd_deg)*(np.pi/180)
                    elif 90 < self.hd_deg <= 180:
                        yaw = -(self.hd_deg-90)*(np.pi/180)
                    elif 180 < self.hd_deg <= 270:
                        yaw = -(self.hd_deg-90)*(np.pi/180)
                    elif 270 < self.hd_deg <= 360:
                        yaw = (360-self.hd_deg+90)*(np.pi/180)

                    quat = tf.transformations.quaternion_from_euler(0, 0, yaw)

                    # self.pose.header.stamp = current_time
                    # self.pose.header.frame_id = 'odom'
                    # self.pose.pose.position.x = vtp_e
                    # self.pose.pose.position.y = vtp_n
                    # self.pose.pose.position.z = 50
                    # self.pose.pose.orientation.x = quat[0]
                    # self.pose.pose.orientation.y = quat[1]
                    # self.pose.pose.orientation.z = quat[2]
                    # self.pose.pose.orientation.w = quat[3]
                    # # path setting
                    # self.path_record.header.frame_id = 'odom'
                    # self.path_record.header.stamp = current_time
                    # self.path_record.poses.append(self.pose)
                    # # number of path 
                    # if len(self.path_record.poses) > 500:
                    #     self.path_record.poses.pop(0)
                    # self.path_pub.publish(self.path_record)
    
                    x = self.gps_pose[0]
                    y = self.gps_pose[1]
                    # x = est_list[-1][1]
                    # y = est_list[-1][0]
                    delta_x = (vtp_e-x)/5
                    delta_y = (vtp_n-y)/5
                    # print('x =', x)
                    # print('est_list[-1][1] =', est_list[-1][1])

                    #x = est_list[-1][1]+r_optimal*np.cos(theta)
                    theta = np.arccos((x-est_list[-1][1])/r_optimal)
                    # print('theta =')
                    # print(theta)

                    alpha = np.arccos((vtp_e-est_list[-1][1])/r_optimal)
                    # print('alpha =')
                    # print(alpha)

                    delta_a = (alpha-theta)/5
                    # print('delta_a =')
                    # print(delta_a)


                    for i in range(5):
                        # path_e = x + delta_x*i
                        # path_n = y + delta_y*i

                        path_e = est_list[-1][1] + r_optimal*np.sin((self.heading-np.pi/2)+12*i*np.pi/180)
                        path_n = est_list[-1][0] + r_optimal*np.cos((self.heading-np.pi/2)+12*i*np.pi/180)

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
                        if len(self.path_record.poses) > 20:
                            self.path_record.poses.pop(0)
                        self.path_pub.publish(self.path_record)

                    ########################################

                    endtime = datetime.datetime.now()
                    print ('Total time =')
                    print (endtime - starttime).seconds

            elif  len(azimuth) < 2: # No.3 observation
                
                if len(azimuth) == 0:
                    if [self.u, self.v] != [0, 0]:
                        angle_a_w, angle_e_w, est_ned, est_vector, ob_point, roll_limit = self.cal_aoa_info()
                        azimuth.append(angle_a_w)
                        elevation.append(angle_e_w)
                        ob_points.append(ob_point)
                        vector.append(est_vector)
                        est_list.append(est_ned)
                        uav_pose.append([self.roll, self.pitch, self.yaw])
                        est_list.append(est_ned)
                        
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
                        est_list.append(est_ned)
                        rolllimit.append(roll_limit)
                        uav_pose.append([self.roll, self.pitch, self.yaw])
                        Est_n,Est_e,Est_d = LQ.LeastQ_m(ob_points, azimuth)
                        Est_position = [Est_n,Est_e,Est_d]
                        est_list.append(Est_position)

                    ### Go to No.2 observation point ###
                    k = 40
                    sin_gamma = (pro_vector[0]*est_vector[1]-pro_vector[1]*est_vector[0])/(np.sqrt(np.square(pro_vector[0])+np.square(pro_vector[1]))*np.sqrt(np.square(est_vector[0])+np.square(est_vector[1])))
                    sin_gamma = np.abs(sin_gamma)
                    acc_cmd = np.square(self.vel)/50 + k*sin_gamma
                    hd_rate = acc_cmd/self.vel
                    print('-------------- ')
                    print('sin_gamma = ')
                    print(sin_gamma*180/np.pi)
                    # print('-------------- ')
                    # print('uav velocity = ')
                    # print(self.vel)
                    # print('acc_cmd = ')
                    # print(acc_cmd)
                    print('hd_rate = ')
                    print(hd_rate)
                    yaw_rate_list.append(hd_rate)
                    sin_list.append(sin_gamma)    

                    if (self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - self.last_req) > rospy.Duration(0.5)):
                        offb_set_mode = SetModeRequest()
                        offb_set_mode.custom_mode = 'OFFBOARD'
                        if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
                            rospy.loginfo("OFFBOARD enabled")
                        self.last_req = rospy.Time.now()
                    else:
                        #self.setpoint.type_mask = 0b01111111011
                        self.setpoint.position.z = 50   
                        # self.attitude.body_rate.z = hd_rate
                        self.setpoint.yaw_rate = hd_rate
                        self.setpoint_pub.publish(self.setpoint)
                        #self.attitude_pub.publish(self.attitude)
            else:
                print('=== Mission finished ===')
                                 
                n,e = est_list[-1][0],est_list[-1][1]
                print('------------------------')
                print('final estimated target position = ')
                print(n,e)
                est = [e, n, 0]

                ## distance between uav and estimated target
                distance = np.sqrt(np.square(self.gps_pose[1]-n)+np.square(self.gps_pose[0]-e))

                if distance >= 50 :
                    r_optimal = distance
                else:
                    r_optimal = 50

                ## waypoint Control ##
                L = distance
                p = self.gps_pose
                vtp_1, vtp_2 = self.insec(p, L, est, L)
                
                uav_to_vtp_1 = np.arctan2(vtp_1[0]-p[0], vtp_1[1]-p[1]) 
                uav_to_vtp_2 = np.arctan2(vtp_2[0]-p[0], vtp_2[1]-p[1]) 
                angle_1 = self.heading - uav_to_vtp_1
                angle_2 = self.heading - uav_to_vtp_2


                if self.heading > 0:
                    if np.abs(angle_1) < np.abs(angle_2):
                        vtp_n = vtp_1[1]
                        vtp_e = vtp_1[0]
                    else:
                        vtp_n = vtp_2[1]
                        vtp_e = vtp_2[0]
                else:
                    if angle_1 > angle_2:
                        vtp_n = vtp_1[1]
                        vtp_e = vtp_1[0]
                    else:
                        vtp_n = vtp_2[1]
                        vtp_e = vtp_2[0]

                if (self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - self.last_req) > rospy.Duration(0.5)):
                    offb_set_mode = SetModeRequest()
                    offb_set_mode.custom_mode = 'OFFBOARD'
                    if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
                        rospy.loginfo("OFFBOARD enabled")
            
                    self.last_req = rospy.Time.now()
                else:
                    self.setpoint.position.x = vtp_e
                    self.setpoint.position.y = vtp_n
                    self.setpoint.position.z = 50
                    self.setpoint_pub.publish(self.setpoint)

        else:
            if len(est_list) == 0:
                print('=== No target!! ===')

            elif len(est_list) <= 2:
                pass

            elif len(est_list) > 2 and len(azimuth) <= 10:
                print('=== Target dispears!! ===')

                ## Go back to last eastimated target position ## 
                n,e = est_list[-1][0],est_list[-1][1]
                print('------------------------')
                print('est_target position = ')
                print(n,e)
                est = [e, n, 0]

                ## distance between uav and estimated target
                distance = np.sqrt(np.square(self.gps_pose[1]-n)+np.square(self.gps_pose[0]-e))

                ### step1 : find optimal circle radius ###
                #r_optimal = self.pathplanning(est_list[-1], distance, rolllimit, hd)
                # if self.roll < rolllimit : 
                if distance >= 50 :
                    r_optimal = distance
                else:
                    r_optimal = 50

                print('r_optimal =')
                print(r_optimal)
                ### step2 : Go to the optimal circle ###
                ## waypoint Control ##
                L = distance
                p = self.gps_pose
                vtp_1, vtp_2 = self.insec(p, L, est, L)
                
                uav_to_vtp_1 = np.arctan2(vtp_1[0]-p[0], vtp_1[1]-p[1]) 
                uav_to_vtp_2 = np.arctan2(vtp_2[0]-p[0], vtp_2[1]-p[1]) 
                # print('----------------------')
                # print('uav_to_vtp_1, uav_to_vtp_2 = ')
                # print(uav_to_vtp_1, uav_to_vtp_2)
                # print('self.heading = ')
                # print(self.heading)

                angle_1 = self.heading - uav_to_vtp_1
                angle_2 = self.heading - uav_to_vtp_2
                # print('angle_1, angle_2 = ')
                # print(angle_1, angle_2)

                if self.heading > 0:
                    if np.abs(angle_1) < np.abs(angle_2):
                        vtp_n = vtp_1[1]
                        vtp_e = vtp_1[0]
                    else:
                        vtp_n = vtp_2[1]
                        vtp_e = vtp_2[0]
                else:
                    if angle_1 > angle_2:
                        vtp_n = vtp_1[1]
                        vtp_e = vtp_1[0]
                    else:
                        vtp_n = vtp_2[1]
                        vtp_e = vtp_2[0]
                print('----------------------')
                print('vtp_e, vtp_n = ')
                print(vtp_e, vtp_n)
                print('----------------------')

                vtp.append([vtp_e, vtp_n])

                if (self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - self.last_req) > rospy.Duration(0.5)):
                    offb_set_mode = SetModeRequest()
                    offb_set_mode.custom_mode = 'OFFBOARD'
                    if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
                        rospy.loginfo("OFFBOARD enabled")
            
                    self.last_req = rospy.Time.now()
                else:
                    self.setpoint.position.x = vtp_e
                    self.setpoint.position.y = vtp_n
                    self.setpoint.position.z = 50
                    self.setpoint_pub.publish(self.setpoint)

                    ############# Record ####################
                    current_time = rospy.Time.now()
                    br = tf.TransformBroadcaster()
                    # translate matrix
                    br.sendTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                                    rospy.Time.now(), "odom", "map")

                    if 0 <= self.hd_deg <= 90:
                        yaw = (90- self.hd_deg)*(np.pi/180)
                    elif 90 < self.hd_deg <= 180:
                        yaw = -(self.hd_deg-90)*(np.pi/180)
                    elif 180 < self.hd_deg <= 270:
                        yaw = -(self.hd_deg-90)*(np.pi/180)
                    elif 270 < self.hd_deg <= 360:
                        yaw = (360-self.hd_deg+90)*(np.pi/180)

                    quat = tf.transformations.quaternion_from_euler(0, 0, yaw)

                    # self.pose.header.stamp = current_time
                    # self.pose.header.frame_id = 'odom'
                    # self.pose.pose.position.x = vtp_e
                    # self.pose.pose.position.y = vtp_n
                    # self.pose.pose.position.z = 50
                    # self.pose.pose.orientation.x = quat[0]
                    # self.pose.pose.orientation.y = quat[1]
                    # self.pose.pose.orientation.z = quat[2]
                    # self.pose.pose.orientation.w = quat[3]
                    # # path setting
                    # self.path_record.header.frame_id = 'odom'
                    # self.path_record.header.stamp = current_time
                    # self.path_record.poses.append(self.pose)
                    # # number of path 
                    # if len(self.path_record.poses) > 500:
                    #     self.path_record.poses.pop(0)
                    # self.path_pub.publish(self.path_record)
            
                    x = self.gps_pose[0]
                    y = self.gps_pose[1]
                    # x = est_list[-1][1]
                    # y = est_list[-1][0]
                    delta_x = (vtp_e-x)/5
                    delta_y = (vtp_n-y)/5

                    #x = est_list[-1][1] + r_optimal*np.cos(theta)
                    theta = np.arccos((x-est_list[-1][1])/r_optimal)
                    # print('theta =')
                    # print(theta)

                    alpha = np.arccos((vtp_e-est_list[-1][1])/r_optimal)
                    # print('alpha =')
                    # print(alpha)

                    delta_a = (alpha-theta)/5
                    # print('delta_a =')
                    # print(delta_a)


                    for i in range(5):
                        # path_e = x + delta_x*i
                        # path_n = y + delta_y*i
                        
                        path_e = est_list[-1][1] + r_optimal*np.sin((self.heading-np.pi/2)+12*i*np.pi/180)
                        path_n = est_list[-1][0] + r_optimal*np.cos((self.heading-np.pi/2)+12*i*np.pi/180)
        
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
                        if len(self.path_record.poses) > 20:
                            self.path_record.poses.pop(0)
                        self.path_pub.publish(self.path_record)
                    ########################################

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

