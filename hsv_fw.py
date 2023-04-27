import cv2
import numpy as np

cap = cv2.VideoCapture('20230209_v2.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('hsvoutput.mp4', fourcc, 30.0, (1920, 1080))

c = 1
timeF = 10

while(True):
    ret, frame = cap.read()
    # mask 1
    mask_half = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8)
    mask_half[300:720,0:1280] = 255
    frame_1 = cv2.bitwise_and(frame, frame, mask = mask_half)

    # mask 2
    hsv = cv2.cvtColor(frame_1, cv2.COLOR_BGR2HSV)
    # lower_white = np.array([0,0,130])
    # upper_white = np.array([180,90,255])
    lower_white = np.array([90,10,212])
    upper_white = np.array([131,255,255])
    r_lower = np.array([140, 60, 130])
    r_upper = np.array([190, 140, 255])

    r_mask = cv2.inRange(hsv, lower_white, upper_white)

    kernal = np.ones((25, 25), "uint8")
    r_mask = cv2.dilate(r_mask, kernal)

    frame_2 = cv2.bitwise_and(frame_1, frame_1, mask =r_mask)

    #cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow('mask', cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow('res', cv2.WINDOW_KEEPRATIO)

    contours, hierarchy = cv2.findContours(r_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):

        area = cv2.contourArea(contour)
        if 1000> area > 10:
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            p = x + (w//2)
            q = y + (h//2)
            center = [p, q]
            Target = 'Target:'+str(center)
            cv2.putText(frame, Target, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            #self.center = rospy.center()

            #if (q>80):
            #print('Target', self.center)
            #print("Number of Contours found = " + str(len(contours)))

        cv2.imshow("detection", frame)
        # cv2.imshow('frame',frame)
        out.write(frame)
        # cv2.imshow('mask',mask)
        # cv2.imshow('res',res)
        cv2.waitKey(1)
        if c % timeF == 0:
            cv2.imwrite('hsvoutputfile' + str(int(c / timeF)) + '.jpg', frame)
        c+= 1
    
    if cv2.waitKey(10) == 27:
      break

cap.release()

out.release()

cv2.destroyAllWindows()
