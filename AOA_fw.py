import math
import numpy as np

class AOA:
    def AOA_v0(self, x, y, z, phi, theta, psi, P_img_x, P_img_y, P_img_z):

        ##### Coordinate Transfermation #####
        # step 1 : body frame to camera
        ## YAW turn right 90 degree
        # b2c = [[1, 0, 0],[0, 0, -1],[0, 1, 0]]
        ## Gimbal attitude (tilt,pan,roll)
        roll,tilt,pan = -10*np.pi/180, 90*np.pi/180, 0
        C_x = [[1, 0, 0],[0, np.cos(roll), -np.sin(roll)],[0, np.sin(roll), np.cos(roll)]]
        C_y = [[np.cos(tilt), 0, -np.sin(tilt)], [0, 1, 0], [np.sin(tilt), 0, np.cos(tilt)]]
        C_z = [[np.cos(pan), -np.sin(pan), 0], [np.sin(pan), np.cos(pan), 0], [0, 0, 1]]
        C_1 = np.matmul(C_y,C_z)
        B2C = np.matmul(C_x, C_1)
        C2B = B2W = np.linalg.inv(B2C)
        
        #step 2 : world coordinate to body frame
        ## uav attitude (ned)
        R_x = [[1, 0, 0],[0, np.cos(phi), -np.sin(phi)],[0, np.sin(phi), np.cos(phi)]]
        R_y = [[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]]
        R_z = [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]]
        R_1 = np.matmul(R_y,R_z)
        W2B = np.matmul(R_x, R_1)
        B2W = np.linalg.inv(W2B)

        ############ Calculation ###################
        ## Start : from vector in camera coordinate ##
        P_ct = [[P_img_x], [P_img_y], [P_img_z]]
        norm_T = np.linalg.norm(P_ct)
        P_ct_norm = P_ct/norm_T
        L_r = [0, 0, 1]

        #step 1 : camera to body frame
        P_bt = np.matmul(C2B, P_ct_norm)
        #step 2 : body to world coordinate
        P_wt = np.matmul(B2W, P_bt) # target position vector in world coordinate
        #step 3 : estimate the scalar
        d = abs(z)/(np.dot(L_r, P_wt))
        # print("z = " , z)
        # print("L_s = " , L_s)
        # print("d = " , d)

        ## Method_1 
        P_world = np.matmul(P_wt, d)
        #print("target position vector = ")
        #print(P_world)

        est_n = P_world[0] + x  
        est_e = P_world[1] + y   
        est_d = P_world[2] + z  
 
        ## Method_2 unit vector
        vector_n = P_wt[0]
        vector_e = P_wt[1]
        vector_d = P_wt[2]
    
        ## a = azimuth angle   ##
        ## e = elevation angle ##
        a_w = np.arctan2(est_e-y, est_n-x)
        a = np.arctan2(vector_e, vector_n)

        e_w = np.arctan2(np.sqrt(np.square(est_n - x) + np.square(est_e - y)), est_d - z)
        e = np.arctan2(np.sqrt(np.square(vector_n) + np.square(vector_e)), vector_d)
    
        return a_w, e_w, a, e, est_n, est_e, est_d, vector_n, vector_e, vector_d

    def AOA_v2(self, x, y, z, phi, theta, psi, P_img_x, P_img_y, P_img_z):

        #step 1 : camera to body frame
        C = [[0, 0, 1],[0, -1, 0],[-1, 0, 0]]

        C_x = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
        C_y = [[math.cos(10), 0, -math.sin(10)],[0, 1, 0],[math.sin(10), 0, math.cos(10)]]
        C_z = [[math.cos(90), -math.sin(90), 0],[math.sin(90), math.cos(90), 0],[0, 0, 1]]
        C_C = np.matmul(C_y, C_z)
        #print(C)

        #step 2 : body to world coordinate
        # uav pose
        R_x = [[1, 0, 0],[0, math.cos(phi), -math.sin(phi)],[0, math.sin(phi), math.cos(phi)]]
        R_y = [[math.cos(theta), 0, -math.sin(theta)], [0, 1, 0], [math.sin(theta), 0, math.cos(theta)]]
        R_z = [[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0, 0, 1]]

        R_1 = np.matmul(R_x,R_y)
        R = np.matmul(R_1, R_z)
        #print(R)
        inv_R = np.linalg.inv(R)
        #print("inv_R = " , inv_R)

        ## start ##
        C_T = [[P_img_x], [P_img_y], [P_img_z]]
        norm_T = np.linalg.norm(C_T)
        T_norm = C_T/norm_T
        L_r = [0, 0, 1]
        #print("T = ")
        #print(T)
        #print("norm_T = ")
        #print(norm_T)

        #step 1 : camera to body frame
        C_C = np.matmul(C, T_norm)
        T = np.matmul(inv_R, C_C)
        #step 2 : body to world coordinate
        L_s = np.matmul(inv_R,T) # target position vector in world coordinate
        d = abs(z)/(np.dot(L_r, L_s)) #scalar
        # print("z = " , z)
        # print("L_s = " , L_s)
        # print("d = " , d)

        ## Method_1 
        P_world = np.matmul(L_s, d) #enu
        #print("target position vector = ")
        #print(P_world)

        est_n = P_world[1] + x   #N
        est_e = P_world[0] + y   #E 
        est_d = -P_world[2] + z  #D 
        # est_n = 0   #N
        # est_e = 50   #E 
        # est_d = 0 #D 

        ## Method_2 unit vector
        vector_n = L_s[1]
        vector_e = L_s[0]
        vector_d = -L_s[2]
    
        ## a = azimuth angle   ##
        ## e = elevation angle ##
        a_w = np.arctan2(est_e-y, est_n-x)
        a = np.arctan2(vector_e, vector_n)

        e_w = np.arctan2(np.sqrt(np.square(est_n - x) + np.square(est_e - y)), est_d - z)
        e = np.arctan2(np.sqrt(np.square(vector_n) + np.square(vector_e)), vector_d)
    
        return a_w, e_w, a, e, est_n, est_e, est_d, vector_n, vector_e, vector_d

    def AOA_v3(self, x, y, z, phi, theta, psi, P_img_x, P_img_y, P_img_z):

        #step 1 : camera to body frame
        C = [[0, 0, 1],[0, -1, 0],[-1, 0, 0]]

        C_x = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
        C_y = [[math.cos(10), 0, -math.sin(10)],[0, 1, 0],[math.sin(10), 0, math.cos(10)]]
        C_z = [[math.cos(90), -math.sin(90), 0],[math.sin(90), math.cos(90), 0],[0, 0, 1]]
        C_C = np.matmul(C_y, C_z)
        #print(C)

        #step 2 : body to world coordinate
        # uav pose
        R_x = [[1, 0, 0],[0, math.cos(phi), -math.sin(phi)],[0, math.sin(phi), math.cos(phi)]]
        R_y = [[math.cos(theta), 0, -math.sin(theta)], [0, 1, 0], [math.sin(theta), 0, math.cos(theta)]]
        R_z = [[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0, 0, 1]]

        R_1 = np.matmul(R_x,R_y)
        R = np.matmul(R_1, R_z)
        #print(R)
        inv_R = np.linalg.inv(R)
        #print("inv_R = " , inv_R)

        ## start ##
        C_T = [[P_img_x], [P_img_y], [P_img_z]]
        norm_T = np.linalg.norm(C_T)
        T_norm = C_T/norm_T
        L_r = [0, 0, 1]
        #print("T = ")
        #print(T)
        #print("norm_T = ")
        #print(norm_T)

        #step 1 : camera to body frame
        C_C = np.matmul(C, T_norm)
        T = np.matmul(inv_R, C_C)
        #step 2 : body to world coordinate
        L_s = np.matmul(inv_R,T) # target position vector in world coordinate
        d = abs(z)/(np.dot(L_r, L_s)) #scalar
        # print("z = " , z)
        # print("L_s = " , L_s)
        # print("d = " , d)

        ## Method_1 
        P_world = np.matmul(L_s, d) #enu
        #print("target position vector = ")
        #print(P_world)

        est_n = P_world[1] + x   #N
        est_e = P_world[0] + y   #E 
        est_d = -P_world[2] + z  #D 

        ## Method_2 unit vector
        vector_n = L_s[1]
        vector_e = L_s[0]
        vector_d = -L_s[2]
    
        ## a = azimuth angle   ##
        ## e = elevation angle ##
        a_w = np.arctan2(est_e-y, est_n-x)
        a = np.arctan2(vector_e, vector_n)

        e_w = np.arctan2(np.sqrt(np.square(est_n - x) + np.square(est_e - y)), est_d - z)
        e = np.arctan2(np.sqrt(np.square(vector_n) + np.square(vector_e)), vector_d)
    
        return e_w
