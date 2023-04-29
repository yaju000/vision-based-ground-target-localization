#!/usr/bin/env python
# path following
import math
import numpy as np
import rospy
from pandas import DataFrame
from mavros_msgs.msg import State, AttitudeTarget, Thrust
from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
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
roll_cpmmand = []

class aoa_info(object):

    def __init__(self):
        self.gps_pose = [0,0,0]
        self.ned_pose = [0,0,0]
        self.vel_pose = [0,0,0]
        self.u_u, self.v_v = 0, 0
        self.imu_x = 0
        self.lamda = 0
        self.roll, self.pitch, self.yaw = 0,0,0
        self.roll_rate, self.pitch_rate, self.yaw_rate = 0, 0, 0
        self.quat = [0,0,0,0]
        self.hd_deg = 0
        self.heading = 0
        self.vel = 0
        self.pro_vector = [0, 0]
        self.roll_cmd = 0
        self.last_req = rospy.Time.now()
        ## Subscribe Topic
        rospy.Subscriber("/mavros/state", State, self.state_cb)
        rospy.Subscriber("/mavros/imu/data", Imu, self.imu_callback)
        rospy.Subscriber("/mavros/global_position/local", Odometry, self.gps_callback)    
        rospy.Subscriber("/mavros/global_position/compass_hdg", Float64, self.hdg_callback)
        rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, self.vel_callback)
        ## Ros Service
        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        ## Publish Topic
        # self.setpoint_pub = rospy.Publisher("/plane_cam_0/mavros/setpoint_raw/local", PositionTarget, queue_size=1)
        # self.setpoint = PositionTarget()
        #self.setpoint.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        #self.setpoint.coordinate_frame = PositionTarget.FRAME_BODY_NED
        # Control Pz+YAW RATE
        #self.setpoint.type_mask = 0b0000011111111011
        # Control Velocity + YAW RATE
        #self.setpoint.type_mask = 0b0000011111000111
        # Control Velocity
        # self.setpoint.type_mask = 0b0000111111000111
        # Control Acceleraion + YAW RATE
        #self.setpoint.type_mask = 0b0000111000111111
        # Control Acceleraion + YAW
        #self.setpoint.type_mask = 0b0000111000111111
 
        self.attitude_pub = rospy.Publisher("/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1)
        self.attitude = AttitudeTarget()
        self.attitude.type_mask = 0b0000011111
        # self.attitude.type_mask = IGNORE_ROLL_RATE + IGNORE_PITCH_RATE + IGNORE_YAW_RATE
        # self.path_pub = rospy.Publisher('/plane_cam_0/mavros/setpoint_trajectory/desired', Path, queue_size=50)
        # self.pose = PoseStamped()
        # self.path_record = Path()

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

        self.roll_rate = msg.angular_velocity.x
        self.pitch_rate = msg.angular_velocity.y
        self.yaw_rate = msg.angular_velocity.z

        self.roll, self.pitch, self.yaw = self.euler_from_quaternion(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)

    def hdg_callback(self, msg):
        self.hdg_msg = msg
        heading_angle = msg.data #ned
        self.hd_deg = heading_angle #ned # degree
        self.heading = 90-heading_angle 

        if self.heading <= -180:
            self.heading = self.heading + 360
        else:
            self.heading = self.heading

        self.heading = np.deg2rad(self.heading) #enu
        #print('hd_enu_deg =', self.heading) 

        # if heading_angle > 180:
        #     self.heading = np.deg2rad(heading_angle-360)
        # else:
        #     self.heading = np.deg2rad(heading_angle)

        # print('heading angle(deg) = ')
        # print(heading_angle)
        # print('heading angle(rad) = ')
        # print(self.heading)

    def vel_callback(self, msg): #enu
        self.vel_msg = msg
        self.vel_pose[0] = msg.twist.linear.x
        self.vel_pose[1] = msg.twist.linear.y
        self.vel_pose[2] = msg.twist.linear.z
        self.vel = np.sqrt(self.vel_pose[0]**2+self.vel_pose[1]**2)
        # print('self.vel_pose')
        # print(np.sqrt(self.vel_pose[0]**2+self.vel_pose[1]**2))

    def euler_from_quaternion(self, x, y, z, w): #enu
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

    def angle(self, v1, v2):
        angle1 = math.atan2(v1[0], v1[1])
        angle1 = int(angle1 * 180/math.pi)
        # print(angle1)
        angle2 = math.atan2(v2[0], v2[1])
        angle2 = int(angle2 * 180/math.pi)
        # print(angle2)
        if angle1*angle2 >= 0:
            included_angle = abs(angle1-angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle

    def angle_of_vector(self, v1, v2):
        pi = 3.1415
        vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
        length_prod = sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * sqrt(pow(v2[0], 2) + pow(v2[1], 2))
        cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
        return (acos(cos) / pi) * 180

    def height_control(self, pos, vel, K):

        h_error = 50-pos[2]
        pitch_d = (h_error/vel)*K
        vz = (h_error)*K
   
        return pitch_d, vz

    def position_control(self, pose, vtp, kp):

        dt = 1/20.0
        e_error = vtp[0] - pose[0]
        n_error = vtp[1] - pose[1]

        ve = kp*e_error/5
        vn = kp*n_error/5

        desired_ve = self.vel_pose[0]-ve
        desired_vn = self.vel_pose[1]-vn
        return ve, vn

    def ENU2BODY(self, a, b, c, body_matrix):
        R_x = [[1, 0, 0],[0, np.cos(a), -np.sin(a)],[0, np.sin(a), np.cos(a)]]
        R_y = [[np.cos(b), 0, -np.sin(b)], [0, 1, 0], [np.sin(b), 0, np.cos(b)]]
        R_z = [[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]]

        R_1 = np.matmul(R_y,R_z)
        R = np.matmul(R_x, R_1)
        inv_R = np.linalg.inv(R)

        world_matrix = np.matmul(inv_R, body_matrix)
        return world_matrix

    def iteration(self, event):

        print('--------------------------------')
        print('roll, pitch, yaw = ')
        print(self.roll, self.pitch, self.yaw)
        print('world velocity = ')
        print(self.vel_pose)
        print('--------------------------------')

        ################ L1 Nonlinear #####################
        ############### Straight-line #####################
        # L = 50
        # #### determin the vtp ####
        # vtp_e = np.sqrt(np.square(L)-np.square(self.gps_pose[1]))+self.gps_pose[0]
        # vtp_n = 0
        # vtp = [vtp_e, vtp_n]
        # print('vtp =', vtp)

        # v1 = [self.vel_pose[0], self.vel_pose[1]]
        # v2 = [vtp_e-self.gps_pose[0], vtp_n-self.gps_pose[1]]
        # i = self.angle_of_vector(v1, v2)
        # u = 2*np.square(self.vel)*np.sin(i)/L
        # print('v1, v2 =', v1, v2)
        # print('i =', i)
        # print('u =', u)
        #################### Circle #######################
        Radius = 100
        L = 50
        est = [0, 0, 0]

        distance = np.sqrt(np.square(self.gps_pose[1]-est[1])+np.square(self.gps_pose[0]-est[0]))
 
        if distance >= L :
            r_optimal = distance
        else:
            r_optimal = L

        p = self.gps_pose
        vtp_1, vtp_2 = self.insec(p, L, est, Radius)

        if [vtp_1, vtp_2] == [None, None]:
            vtp_1, vtp_2 = self.insec(p, r_optimal, est, r_optimal)
        else:
            pass
        
        # uav_to_vtp_1 = np.arctan2(vtp_1[0]-p[0], vtp_1[1]-p[1]) #NED
        # uav_to_vtp_2 = np.arctan2(vtp_2[0]-p[0], vtp_2[1]-p[1]) 
        uav_to_vtp_1 = np.arctan2(vtp_1[1]-p[1], vtp_1[0]-p[0]) #ENU
        uav_to_vtp_2 = np.arctan2(vtp_2[1]-p[1], vtp_2[0]-p[0]) 
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
        #### parameter setting ####
        circle_error = distance - Radius
        #desired_yaw = np.arctan2(vtp_e-p[0], vtp_n-p[1]) # NED
        desired_yaw = np.arctan2(vtp_n-p[1], vtp_e-p[0])  # ENU
        gamma = self.heading - desired_yaw
        vel = np.sqrt(self.vel_pose[0]**2+self.vel_pose[1]**2)
        #print('velocity', vel)

        ####### attitude control #######
        u = 2*np.square(vel)*np.sin(gamma)/L + 0.05*circle_error

        #roll_cmd = np.abs(np.arctan(u/9.81)) #ENU
        roll_cmd = np.arctan(u/9.81)
        yaw_rate = u/vel
        yaw_cmd = self.heading
        pitch_cmd, vz = self.height_control(self.gps_pose, vel, 0.01)
        pitch_cmd = pitch_cmd*np.pi/180

        if roll_cmd > 0.785:
            roll_cmd = 0.785
        elif roll_cmd < -0.785:
            roll_cmd = -0.785
        else:
            roll_cmd = roll_cmd

        if pitch_cmd > 0.157:
            pitch_cmd = 0.157
        elif pitch_cmd < -0.157:
            pitch_cmd = -0.157
        else:
            pitch_cmd = pitch_cmd

        print('circle_error =', circle_error)
        print('u =',u)
        #print('gamma =',gamma)
        print('roll_cmd =', roll_cmd)
        #print('yaw_rate =', yaw_rate)
        print('pitch_cmd =', pitch_cmd)
        print('yaw_cmd =', yaw_cmd)
        #print('r_optimal =', r_optimal)
        #print('heading angle(rad)enu = ', self.heading)
        ##############################

        # ax, ay, az = self.ENU2BODY(self.roll, self.pitch, self.yaw, [[u],[0],[0]]) 
        # print('----------------------')
        # print('ax, ay, az =', ax, ay, az)
        # print('----------------------')
        # print('yaw angle =', self.yaw)
        # print('yaw angle(deg) =', self.yaw*180/np.pi)
        # print('desired_yaw =', desired_yaw)
        # print('desired_yaw(deg) =', desired_yaw*180/np.pi)
        # print('gamma = ', gamma)
        # print('gamma(deg) = ', gamma*180/np.pi)
        # print('u =', u)
        # print('----------------------')
       
        ############## Position control####################
        # vtp = [vtp_e, vtp_n]
        # ve, vn = self.position_control(self.gps_pose, vtp, 2.0)
       
        # #vb = [[15], [0], [-10]]
        # vw = [[ve], [vn], [vz]]
        # #vw = [15, 15, vz]
    
        # # ENU2BODY
        # #vw = self.b2w(self.roll, self.pitch, self.yaw, vb)
        # print('velocity(e,n,u)=', vw) #enu
        # #print('Body velocity =', vb)

        ###################################################
        if (self.current_state.mode != "GUIDED" and (rospy.Time.now() - self.last_req) > rospy.Duration(1.0/20)):
            pass
            # offb_set_mode = SetModeRequest()
            # offb_set_mode.custom_mode = 'GUIDED'
            # if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
            #     rospy.loginfo("GUIDED enabled")
            # self.last_req = rospy.Time.now()
        else:
            #pass
            #print('loiter to estimated point')
            ### /mavros/setpoint_raw/attitude ###
            ## 1: roll control
            #quat = tf.transformations.quaternion_from_euler(0, pitch_cmd, yaw_cmd) # roll pitch yaw
            quat = tf.transformations.quaternion_from_euler(roll_cmd, pitch_cmd, 0) # roll pitch yaw
            self.attitude.orientation.x = quat[0]
            self.attitude.orientation.y = quat[1]
            self.attitude.orientation.z = quat[2]
            self.attitude.orientation.w = quat[3]
            self.attitude.thrust = 0.7
            self.attitude_pub.publish(self.attitude)
            ## 2: body rate control 
            # self.attitude.type_mask = 0b00000000
            # self.attitude.type_mask = 0b11111000
            # self.attitude.body_rate.x = 0.087
            # self.attitude.body_rate.y = 0
            # self.attitude.body_rate.z = 0
            # self.attitude_pub.publish(self.attitude)

            ### /mavros/setpoint_raw/local ###
            ## 1: heading rate control
            # self.setpoint.position.x = vtp_e
            # self.setpoint.position.y = vtp_n
            # self.setpoint.position.z = 50
            # self.setpoint.velocity.x = 13
            # self.setpoint.velocity.y = 0
            # self.setpoint.velocity.z = 0
            # self.setpoint.acceleration_or_force.x = ax
            # self.setpoint.acceleration_or_force.y = ay
            # self.setpoint.acceleration_or_force.z = az
            # self.setpoint.yaw = desired_yaw
            # self.setpoint.yaw_rate = yaw_rate
            #self.setpoint_pub.publish(self.setpoint)
   

if __name__ == '__main__':
    
    rospy.init_node('pathfollowing_py', anonymous=True)
    dt = 1.0/20
    #dt = 1.0/5
    pathplan_run = aoa_info()
    rospy.Timer(rospy.Duration(dt), pathplan_run.iteration)
    rospy.spin()

    df = DataFrame({'enu_pos': enu_pos})
    df.to_excel('fw_pathfollowing.xlsx', sheet_name='sheet1', index=False)
