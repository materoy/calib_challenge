# ! /usr/bin/python3

import numpy as np
import cv2 as cv

cap = cv.VideoCapture('unlabeled/9.hevc')
orb = cv.ORB_create()

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Ok the video is over now let's watch superman")
        print("Exiting ...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    
    # find the keypoints with ORB
    kp = orb.detect(frame)
    # kp, des = orb.detectAndCompute(gray,None)

    # compute the descriptors with ORB
    kp, des = orb.compute(frame, kp)

    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(frame, kp, frame,color=(0, 255, 0), flags=0)


    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
