#!/usr/bin/env python
# ground target tracking
import math
from math import sqrt, pow, acos
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from itertools import combinations
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

e_pos = []
n_pos = []
u_pos = []
azimuth = []
elevation = []
ob_points = []
uav_pose = []
vector = []
est_list = []
RMSE_list = []
GDOP_list = []
hd_list = []
ddd = []

Pimg_u = []
Pimg_v = []
roll_angle = []
rollcmd = []
gamma_deg = []

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
        self.tracking_radius = 150
        self.roll, self.pitch, self.yaw = 0,0,0
        self.roll_n, self.pitch_e, self.yaw_d = 0, 0, 0
        self.roll_rate, self.pitch_rate, self.yaw_rate = 0, 0, 0
        self.quat = [0,0,0,0]
        self.u, self.v = 0, 0
        self.bb_area = 0
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
        self.roll_limit = [0, 0]
        self.mode = 0
        self.dd = 0
        self.last_req = rospy.Time.now()

        ############# Mode #################
        # 0 : Searching mode predefined path
        # 1 : Tracking mode-Tracking strategy
        # 2 : Tracking mode-Path planning
        # 3 : Mission Finished
        ####################################

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
   
    def state_cb(self, msg):
        self.current_state = msg

    def gps_callback(self, msg): #enu
        self.gps_msg = msg
        self.gps_pose[0] = msg.pose.pose.position.x
        self.gps_pose[1] = msg.pose.pose.position.y
        self.gps_pose[2] = msg.pose.pose.position.z

        self.ned_pose[0], self.ned_pose[1], self.ned_pose[2] = self.gps_pose[1], self.gps_pose[0], -self.gps_pose[2]
        e_pos.append(self.gps_pose[0])
        n_pos.append(self.gps_pose[1])
        u_pos.append(self.gps_pose[2])

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
        roll_angle.append(self.roll)

    def hdg_callback(self, msg): #enu # pi~(-pi)
        self.hdg_msg = msg
        heading_angle = msg.data #ned
        self.hd_deg = round(heading_angle,0) #ned # degree
        self.heading = 90-heading_angle 

        if self.heading <= -180:
            self.heading = self.heading + 360
        else:
            self.heading = self.heading

        self.heading = np.deg2rad(self.heading) #enu

        # print('heading angle(deg) = ')
        # print(self.hd_deg)
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

        if [self.u, self.v] != [0, 0]:
            ## Position_vector ##
            size_u = 640
            size_v = 360
            u_0 = size_u/2
            v_0 = size_v/2
            # focal length
            f = 277.191356
            #f = 554.382712
            #f = 1108.765424 

            self.P_img_x = u_0 - self.u
            self.P_img_y = v_0 - self.v
            self.P_img_z = f

            # self.P_img_x = v_0 - self.v_v
            # self.P_img_y = self.u_u - u_0 
            # self.P_img_z = f
            # P_img = [self.P_img_x, self.P_img_y, self.P_img_z] 
            # print("u, v = ")
            # print(self.u, self.v)
            # print('------------')
            # print("P_img = ")
            # print(P_img)

            self.angle_a_w, self.angle_e_w, self.angle_a, self.angle_e, self.est_n, self.est_e, self.est_d, self.est_vector_n, self.est_vector_e, self.est_vector_d, self.dd = AOA.AOA_v0(self.ned_pose[0], self.ned_pose[1], self.ned_pose[2], self.roll, self.pitch, self.yaw, self.P_img_x, self.P_img_y, self.P_img_z)
            angle_a_w = self.angle_a_w
            angle_e_w = self.angle_e_w
            est_ned = [self.est_n, self.est_e, self.est_d]
            est_vector = [self.est_vector_n, self.est_vector_e, self.est_vector_d]
            ob_point = [self.ned_pose[0], self.ned_pose[1], self.ned_pose[2]]
            dd = self.dd 
            Pimg_u.append(self.P_img_x)
            Pimg_v.append(self.P_img_y)

            return angle_a_w, angle_e_w, est_ned, est_vector, ob_point, dd

    def visual_r(self, r, v, hd, ob, roll, pitch, yaw):

        # focal length
        f = 277.191356
        P_img_c = [0,0,f]

        angle_a_w0, angle_e_w0, angle_a0, angle_e0, est_n, est_e, est_d, cv_n, cv_e, cv_d, dd_0 = AOA.AOA_v0(ob[0], ob[1], ob[2], roll, pitch, yaw, P_img_c[0], P_img_c[1], P_img_c[2])
        pro_vector_0 = [cv_n, cv_e, cv_d]
        ### R_ref ###
        R_ref = dd_0
        acc = np.square(v)/R_ref
        # print('------------------------')
        print('R_ref =', R_ref)
        # print('acc = ', acc)
        # print('acc angle =',hd+90)
        # print('velocity =',v)
        # print('------------------------')
        an = np.deg2rad(hd+90)
        cx_n = acc*np.cos(an)
        cx_e = acc*np.sin(an)
        cx = np.sqrt(np.square(cx_n)+np.square(cx_e))
        c_n = np.abs(cx_n/cx)
        c_e = np.abs(cx_e/cx)
        # print('c_n = ', c_n)
        # print('c_e = ', c_e)
        if 0<=hd<90:
            pro_vector = [-c_n, c_e]
        elif 90<=hd<180:
            pro_vector = [-c_n, -c_e]
        elif 180<=hd<270:
            pro_vector = [c_n, -c_e]  
        elif 270<=hd<=360:
            pro_vector = [c_n, c_e]

        return angle_a_w0, angle_e_w0, pro_vector, pro_vector_0

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

    def height_control(self, pos, vel, K):

        h_error = 70-pos[2]
        pitch_d = (h_error/vel)*K
   
        return pitch_d

    def cal_vtp(self, distance, p, Radius, est, hd_enu):

        L1 = 70

        q = [est[1], est[0], 0]

        if distance >= L1 :
            L1_0 = distance
        else:
            L1_0 = L1

        vtp_1, vtp_2 = self.insec(p, L1, q, Radius)

        # print('----------------------')
        # print('vtp_1, vtp_2 = ')
        # print(vtp_1, vtp_2)
        # print('----------------------')

        if [vtp_1, vtp_2] == [None, None]:
            vtp_1, vtp_2 = self.insec(p, L1_0, q, L1_0)
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
        # print('----------------------')
        # print('vtp_e, vtp_n = ')
        # print(vtp_e, vtp_n)
        # print('----------------------')

        return vtp_e, vtp_n

    def Attitude_control(self, distance, Radius,vtp_e, vtp_n, hd_enu, vel, p):

        L1 = 70
        circle_error = distance - Radius
        desired_yaw = np.arctan2(vtp_n-p[1], vtp_e-p[0])  # ENU
        gamma = hd_enu - desired_yaw

        ####### Accerelation term #######
        u = 2*np.square(vel)*np.sin(gamma)/L1 + 0.01*circle_error

        roll_cmd = np.arctan(u/9.81)
        pitch_cmd = self.height_control(p, vel, 0.05)
        pitch_cmd = pitch_cmd*np.pi/180

        ###### new #######
        # if roll_cmd < self.roll_limit[0]:
        #     roll_cmd = self.roll_limit[0]
        # elif roll_cmd > self.roll_limit[1]:
        #     roll_cmd = self.roll_limit[1]

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

        #print('u =',u)
        # print('circle_error =', circle_error)
        # print('roll_cmd =', roll_cmd)
        # print('pitch_cmd =', pitch_cmd)

        return circle_error, u, roll_cmd, pitch_cmd

    def cal_dop(self,a,p,d):  
        # a = ob_points
        # b = next ob_point 
        # d = estimated position
        # print('--------------------')
        H = []
        H_new = []

        Node_number = len(a)

        for i in range(Node_number):

            r = np.square(d[0]-a[i][0])+np.square(d[1]-a[i][1])
            H_i = [-(d[1]-a[i][1])/r, (d[0]-a[i][0])/r] #1*2
            H.append(H_i)
            H_new.append(H_i)

        r_new = np.square(d[0]-p[0])+np.square(d[1]-p[1])
        HH = [-(d[1]-p[1])/r_new, (d[0]-p[0])/r_new] #1*2

        H_new.append(HH)
        
        QQ = np.matmul(np.transpose(H), H) #2*2
        Q = np.linalg.inv(QQ)
        GDOP = np.sqrt(np.trace(Q))

        QQ_new = np.matmul(np.transpose(H_new), H_new) #2*2
        Q_new = np.linalg.inv(QQ_new)
        GDOP_new = np.sqrt(np.trace(Q_new))
        
        #print("GDOP = ",GDOP)
        #print('--------------------')
        return GDOP, GDOP_new

    def angle_of_vector(self, v1, v2):
        pi = np.pi
        vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
        length_prod = np.sqrt(math.pow(v1[0], 2) + math.pow(v1[1], 2)) * np.sqrt(math.pow(v2[0], 2) + math.pow(v2[1], 2))
        cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
        gamma = (math.acos(cos) / pi) * 180
        return gamma

    def iteration(self, event):

        self.u, self.v, self.bb_area = hsv.value_callback()
        print('=========================')
        print('mode =', self.mode)
        print('(u,v)', self.u, self.v)
        print('bb_area', self.bb_area)
        print('==================')

        if self.mode == 0 : 
            if ([self.u, self.v]!=[None, None]) and ([self.u, self.v]!=[0, 0]):
                
                if len(hd_list) == 0:
                    hd_list.append(self.hd_deg)

                angle_a_w, angle_e_w, est_ned, est_vector, ob_point, dd = self.cal_aoa_info()
                angle_a_w = round(angle_a_w,2)
                if angle_a_w >1.57 :
                    angle_a_w = angle_a_w - 3.14
                elif angle_a_w < -1.57:
                    angle_a_w = 3.14 + angle_a_w

                print('=== Image successful!===')
                print('Current azimuth =', angle_a_w)
                print('Number of observation points =', len(azimuth))
                print('len(est_list) =', len(est_list))

                if est_ned[2] == 0:
                    azimuth.append(angle_a_w)
                    elevation.append(angle_e_w)
                    ob_points.append(ob_point)
                    vector.append(est_vector)
                    uav_pose.append([self.roll, self.pitch, self.yaw])
                    ddd.append(dd)
                    
                if len(azimuth) >= 3:

                    Est_n,Est_e,Est_d = LQ.LeastQ_m(ob_points, azimuth)
                    Est_position = [Est_n,Est_e,Est_d]
                    
                    GDOP, GDOP_new = self.cal_dop(ob_points, self.gps_pose, [0, 50])
                    GDOP_list.append(GDOP)
                    print(np.abs(self.hd_deg-hd_list[0]))
                    if np.abs(self.hd_deg-hd_list[0]) >= 270: 
                        print('Get the initial estimated position!')
                        Est_n,Est_e,Est_d = LQ.LeastQ_m(ob_points, azimuth)
                        Est_position = [Est_n, Est_e, Est_d]
                        est_list.append(Est_position)

                        self.mode = 2

                ### Go to No.2 observation point ###
                angle_a_w0, angle_e_w0, pro_vector,  pro_vector_0 = self.visual_r(150,self.vel,self.hd_deg, ob_point, self.roll, self.pitch, self.yaw)
                est_vector = [self.est_vector_n, self.est_vector_e]
                print('angle_a_w0 =', angle_a_w0)
                print('angle_a_w =', angle_a_w)
                print('pro_vector =', pro_vector_0)
                print('est_vector =', est_vector)

                k = 15
                # sin_gamma = (pro_vector[0]*est_vector[1]-pro_vector[1]*est_vector[0])/(np.sqrt(np.square(pro_vector[0])+np.square(pro_vector[1]))*np.sqrt(np.square(est_vector[0])+np.square(est_vector[1])))
                sin_gamma = (pro_vector_0[0]*est_vector[1]-pro_vector_0[1]*est_vector[0])/(np.sqrt(np.square(pro_vector_0[0])+np.square(pro_vector_0[1]))*np.sqrt(np.square(est_vector[0])+np.square(est_vector[1])))
                # gamma = angle_a_w - angle_a_w0
                e = np.rad2deg(angle_e_w - angle_e_w0)
                
                # gamma = self.angle_of_vector(est_vector, pro_vector_0)
                # print('gamma =', gamma)
                # sin_gamma = np.sin(gamma)
                gamma = np.arcsin(sin_gamma)*180/np.pi
                gamma_deg.append(gamma)

                r = 5
                R = 150
                Rref = R + r*(-e)
                # Rref = 150

                acc_cmd = np.square(self.vel)/Rref + k*sin_gamma
                roll_cmd = np.arctan(acc_cmd/9.81)
                pitch_cmd = self.height_control(self.gps_pose, self.vel, 0.5)
                yaw_cmd = np.deg2rad(gamma)

                if roll_cmd > 0.61:
                    roll_cmd = 0.61
                elif roll_cmd < 0:
                    roll_cmd = 0

                if pitch_cmd > 0.174:
                    pitch_cmd = 0.174
                elif pitch_cmd < -0.174:
                    pitch_cmd = -0.174
        
                print('------visual servo-----------')
                print('e =', e)
                print('Rref =', Rref)
                print('gamma =', gamma)
                print('sin_gamma =', sin_gamma)
                print('acc_cmd =', acc_cmd)
                print('roll_cmd =', roll_cmd)
                print('pitch_cmd =', pitch_cmd)
                print('yaw_cmd =', yaw_cmd)
                print('-----------------------------')

                if (self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - self.last_req) > rospy.Duration(0.5)):
                    offb_set_mode = SetModeRequest()
                    offb_set_mode.custom_mode = 'OFFBOARD'
                    if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
                        rospy.loginfo("OFFBOARD enabled")
                    self.last_req = rospy.Time.now()
                else:
                    quat = tf.transformations.quaternion_from_euler(roll_cmd, -pitch_cmd, yaw_cmd) # roll pitch yaw
                    self.attitude.orientation.x = quat[0]
                    self.attitude.orientation.y = quat[1]
                    self.attitude.orientation.z = quat[2]
                    self.attitude.orientation.w = quat[3]
                    self.attitude.thrust = 0.4
                    self.attitude_pub.publish(self.attitude)
                    rollcmd.append(roll_cmd)
                   
            else:
                if len(azimuth) == 0:
                    print('=== Searching Mode ===')
                    print('=== No target!! ===')
                
                else:

                    print('=== Target dispears!! ===')
                    angle_a_w0, angle_e_w0, pro_vector,  pro_vector_0 = self.visual_r(150,self.vel,self.hd_deg, [self.gps_pose[1], self.gps_pose[0], -self.gps_pose[2]], self.roll, self.pitch, self.yaw)
                    est_vector = [self.est_vector_n, self.est_vector_e]

                    k = 15
                    # sin_gamma = (pro_vector[0]*est_vector[1]-pro_vector[1]*est_vector[0])/(np.sqrt(np.square(pro_vector[0])+np.square(pro_vector[1]))*np.sqrt(np.square(est_vector[0])+np.square(est_vector[1])))
                    sin_gamma = (pro_vector_0[0]*est_vector[1]-pro_vector_0[1]*est_vector[0])/(np.sqrt(np.square(pro_vector_0[0])+np.square(pro_vector_0[1]))*np.sqrt(np.square(est_vector[0])+np.square(est_vector[1])))
                    
                    # gamma = self.angle_of_vector(est_vector, pro_vector_0)
                    # print('gamma =', gamma)
                    # sin_gamma = np.sin(gamma)
                    gamma = np.arcsin(sin_gamma)*180/np.pi
                    gamma_deg.append(gamma)

                    e = np.rad2deg(self.angle_e_w - angle_e_w0)

                    r = 0.5
                    R = 150
                    Rref = R + r*(-e)
                    # Rref = 150
                    
                    acc_cmd = np.square(self.vel)/Rref + k*sin_gamma
                    roll_cmd = np.arctan(acc_cmd/9.81)
                    pitch_cmd = self.height_control(self.gps_pose, self.vel, 0.5)
                    yaw_cmd = np.deg2rad(gamma)

                    if roll_cmd > 0.61:
                        roll_cmd = 0.61
                    elif roll_cmd < 0:
                        roll_cmd = 0

                    if pitch_cmd > 0.174:
                        pitch_cmd = 0.174
                    elif pitch_cmd < -0.174:
                        pitch_cmd = -0.174
            
                    print('------visual servo-----------')
                    print('e =', e)
                    print('Rref =', Rref)
                    print('gamma =', gamma)
                    print('sin_gamma =', sin_gamma)
                    print('acc_cmd =', acc_cmd)
                    print('roll_cmd =', roll_cmd)
                    print('pitch_cmd =', pitch_cmd)
                    print('yaw_cmd =', yaw_cmd)
                    print('-----------------------------')

                    if (self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - self.last_req) > rospy.Duration(0.5)):
                        offb_set_mode = SetModeRequest()
                        offb_set_mode.custom_mode = 'OFFBOARD'
                        if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
                            rospy.loginfo("OFFBOARD enabled")
                        self.last_req = rospy.Time.now()
                    else:
                        quat = tf.transformations.quaternion_from_euler(roll_cmd, -pitch_cmd, yaw_cmd) # roll pitch yaw
                        self.attitude.orientation.x = quat[0]
                        self.attitude.orientation.y = quat[1]
                        self.attitude.orientation.z = quat[2]
                        self.attitude.orientation.w = quat[3]
                        self.attitude.thrust = 0.4
                        self.attitude_pub.publish(self.attitude)
                        rollcmd.append(roll_cmd)

        else:
            print('=== Mission finished ===')
            est = [est_list[-1][0],est_list[-1][1],0]
            RMSE = np.sqrt(np.square(est[0]-0)+np.square(est[1]-50))

            print('Final estimated target position = ', est)
            print('RMSE =', RMSE)
            print('Final GDOP =', GDOP_list[-1])

            ## distance between uav and estimated target
            p = self.gps_pose
            g = 9.81
            distance = np.sqrt(np.square(p[1]-est[0])+np.square(p[0]-est[1]))
            R_star = self.dd
            print('distance =', distance)
            print('the searching radius =', R_star)
            print('==========================')

            ### Find the reference point ###
            vtp_e, vtp_n = self.cal_vtp(distance, self.gps_pose, R_star, est, self.heading)
            ### Attitude control ###
            circle_error, u, roll_cmd, pitch_cmd = self.Attitude_control(distance, R_star, vtp_e, vtp_n, self.heading, self.vel, p)

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
                self.attitude.thrust = 0.4
                self.attitude_pub.publish(self.attitude)
                rollcmd.append(roll_cmd)


if __name__ == '__main__':

    rospy.init_node('main_system', anonymous=True)
    dt = 1.0/20
    pathplan_run = aoa_info()
    rospy.Timer(rospy.Duration(dt), pathplan_run.iteration)
    rospy.spin()

    i = []
    for j in range(len(gamma_deg)):
        i.append(j+1)

    k = []  
    for j in range(len(Pimg_u)):
        k.append(j+1)  

    g = []  
    for j in range(len(Pimg_v)):
        g.append(j+1) 

    
    # fig = plt.figure()
    # ax = plt.axes()
    # plt.plot(e_pos, n_pos,'-')
    # plt.plot(50, 0,'*')
    # for j in range(len(ob_points)):
    #     plt.plot(ob_points[j][1], ob_points[j][0],'*')
    # plt.title("uav path")
    # plt.xlabel("E(m)")
    # plt.ylabel("N(m)")
    # plt.legend()
    # plt.grid()
    # plt.show()
    # fig.savefig('uavpath.jpg')

    fig = plt.figure()
    ax = plt.axes()
    plt.plot(i, gamma_deg,'-')
    plt.title("gamma")
    plt.xlabel("i")
    plt.ylabel("gamma")
    plt.legend()
    plt.grid()
    plt.show()
    fig.savefig('gamma.jpg')

    fig = plt.figure()
    ax = plt.axes()
    plt.plot(k, Pimg_u,'-') 
    plt.title("Pimg_u")
    plt.xlabel("i")
    plt.ylabel("Pimg_u")
    plt.legend()
    plt.grid()
    plt.show()
    fig.savefig('Pimg_u.jpg')

    fig = plt.figure()
    ax = plt.axes()
    plt.plot(g, Pimg_v,'-') 
    plt.title("Pimg_v")
    plt.xlabel("i")
    plt.ylabel("Pimg_v")
    plt.legend()
    plt.grid()
    plt.show()
    fig.savefig('Pimg_v.jpg')

    df = DataFrame({'e': e_pos,'n': n_pos,'u': u_pos})
    df.to_excel('fw_tt_path.xlsx', sheet_name='sheet1', index=False)
    dq = DataFrame({'azimuth':azimuth,'ob_points': ob_points,'vector':vector,'ddd':ddd})
    dq.to_excel('fw_tt_result.xlsx', sheet_name='sheet1', index=False)
    dr = DataFrame({'est_list':est_list})
    dr.to_excel('fw_tt_est.xlsx', sheet_name='sheet1', index=False)
    da = DataFrame({'GDOP_list':GDOP_list})
    da.to_excel('fw_tt_dop.xlsx', sheet_name='sheet1', index=False)
    dg = DataFrame({'RMSE_list':RMSE_list})
    dg.to_excel('fw_tt_RMSE.xlsx', sheet_name='sheet1', index=False)

