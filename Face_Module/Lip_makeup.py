from Get_key_point import face_key_point_classification
import numpy as np
import cv2
import gc


class lip_make_up_guide(object):
    def __init__(self,user_face,user_keypoints:face_key_point_classification):
        self.user_face=user_face
        self.user_keypoints=user_keypoints
        self.blue=cv2.imread("D:\\document\\makeup\\faces\\blue1.jpg")


    def lip_pic(self):
        self.left=min(self.user_keypoints.lips_shape()[:,0])
        self.right=max(self.user_keypoints.lips_shape()[:,0])
        self.top=min(self.user_keypoints.lips_shape()[:,1])
        self.bottom=max(self.user_keypoints.lips_shape()[:,1])
        self.lip_img=self.user_face[int(self.top):int(self.bottom +1),int(self.left):int(self.right +1)]
        self.blue=self.blue[int(self.top):int(self.bottom +1),int(self.left):int(self.right +1)]


    def makeup_guide(self):
        self.lip_pic()
        lip_img_gray=cv2.cvtColor(self.lip_img,cv2.COLOR_BGR2GRAY)
        lip_img_gray = cv2.equalizeHist(lip_img_gray)
        _,lip_img_gray=cv2.threshold(lip_img_gray,110,255,cv2.THRESH_BINARY)
        lip_img_gray=cv2.GaussianBlur(lip_img_gray, (3, 3), 0)##高斯过滤
        lip_img_gray=cv2.bitwise_not(lip_img_gray)
        lip_img_gray = cv2.bitwise_and(self.blue,self.blue, mask=lip_img_gray) ##原始提示图片
        hsv=cv2.cvtColor(lip_img_gray,cv2.COLOR_BGR2HSV)
        lower_black=np.array([0,0,0])
        upper_black=np.array([255,255,46])
        mask_black=cv2.inRange(hsv,lower_black,upper_black)
        lip_img_gray[mask_black!=0]=self.lip_img[mask_black!=0]
        
        alpha = 0.7
        beta = 1-alpha
        gamma = 0
        guide = cv2.addWeighted(self.lip_img,alpha,lip_img_gray,beta,gamma)

        self.guid_img=self.user_face.copy()
        self.guid_img[int(self.top):int(self.bottom +1),int(self.left):int(self.right +1)]=guide
        return self.guid_img



        # cv2.imshow("qqq",self.guid_img)




