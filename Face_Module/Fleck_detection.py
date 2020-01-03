import cv2
import numpy as np


class find_skin_fleck(object):
    def __init__(self,face_img):
        self.face_img=face_img
        self.gray = cv2.cvtColor(self.face_img,cv2.COLOR_BGR2GRAY)
    
    def sharpening_processing(self):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
        self.sharped_img = cv2.filter2D(self.gray, -1, kernel=kernel)

        # cv2.Scharr(self.gray, 3, 10, 10)
        # cv2.imshow("custom_blur_demo", self.sharped_img)
    
    def find_fleck(self):
        self.sharpening_processing()
        self.sharped_img = cv2.erode(self.sharped_img, None, iterations=6)
        self.sharped_img = cv2.dilate(self.sharped_img, None, iterations=3)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        self.sharped_img = cv2.filter2D(self.sharped_img, -1, kernel=kernel)
        # cv2.imshow('lll',self.sharped_img)
        # self.canny_img=cv2.Canny(self.sharped_img, 0, 250)
        # cv2.imshow('canny',self.canny_img)
        detector = cv2.SimpleBlobDetector_create()
 
        keypoints = detector.detect(self.sharped_img)
        
        im_with_keypoints = cv2.drawKeypoints(self.face_img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return im_with_keypoints
        # cv2.imshow("Keypoints", im_with_keypoints)
