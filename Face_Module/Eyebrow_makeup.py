from Get_key_point import face_key_point_classification
import numpy as np
import cv2
import gc


class eyebrow_make_up_guide(object):
    def __init__(self,target_face,target_keypoints:face_key_point_classification,user_face,user_keypoints:face_key_point_classification):
        self.target_face=target_face
        self.target_keypoints=target_keypoints
        self.user_face=user_face
        self.user_keypoints=user_keypoints
        self.blue=cv2.imread("D:\\document\\makeup\\faces\\blue1.jpg")
        self.resize_ratio=user_keypoints.face_width() / target_keypoints.face_width()

    def t_left_eyebrow_pic(self):
        left= min(self.target_keypoints.left_eyebrow()[:,0]) 
        top = min(self.target_keypoints.left_eyebrow()[:,1])
        right = max(self.target_keypoints.left_eyebrow()[:,0])
        bottom = ( max(self.target_keypoints.left_eyebrow()[:,1]) + min(self.target_keypoints.left_eye()[:,1])) / 2
        self.target_left_eyebrow_img = self.target_face[int(top):int(bottom+1),int(left):int(right+1)]
        self.target_left_eyebrow_points= list(map( lambda i: np.array([i[0]-left,i[1]-top]),self.target_keypoints.left_eyebrow() ))
        self.target_left_eyebrow_points=np.array(self.target_left_eyebrow_points)
        self.t_left_reference=self.target_left_eyebrow_points[4]

    def t_right_eyebrow_pic(self):
        right= max(self.target_keypoints.right_eyebrow()[:,0])
        top = min(self.target_keypoints.right_eyebrow()[:,1])
        left =min(self.target_keypoints.right_eyebrow()[:,0])
        bottom = ( max(self.target_keypoints.right_eyebrow()[:,1]) + min(self.target_keypoints.right_eye()[:,1])) / 2
        self.target_right_eyebrow_img = self.target_face[int(top):int(bottom+1),int(left):int(right+1)]
        self.target_right_eyebrow_points= list(map( lambda i: np.array([i[0]-left,i[1]-top]),self.target_keypoints.right_eyebrow() ))
        self.target_right_eyebrow_points =np.array(self.target_right_eyebrow_points)
        self.t_right_reference=self.target_right_eyebrow_points[0]



    def makeup_guide(self):
        self.t_left_eyebrow_pic()
        self.t_right_eyebrow_pic()

        self.blue_left = cv2.resize(self.blue,(self.target_left_eyebrow_img.shape[1],self.target_left_eyebrow_img.shape[0]) , interpolation = cv2.INTER_AREA)
        self.blue_right = cv2.resize(self.blue,(self.target_right_eyebrow_img.shape[1],self.target_right_eyebrow_img.shape[0]) , interpolation = cv2.INTER_AREA)
   

        target_left_eyebrow_img_gray =cv2.cvtColor(self.target_left_eyebrow_img,cv2.COLOR_BGR2GRAY)##转为灰度图
        target_right_eyebrow_img_gray =cv2.cvtColor(self.target_right_eyebrow_img,cv2.COLOR_BGR2GRAY)
        
        target_left_eyebrow_img_gray = cv2.equalizeHist(target_left_eyebrow_img_gray)
        target_right_eyebrow_img_gray = cv2.equalizeHist(target_right_eyebrow_img_gray)

        _,target_left_eyebrow_img_gray=cv2.threshold(target_left_eyebrow_img_gray,90,255,cv2.THRESH_BINARY)##提取深色特征
        _,target_right_eyebrow_img_gray=cv2.threshold(target_right_eyebrow_img_gray,90,255,cv2.THRESH_BINARY)
        target_left_eyebrow_img_gray=cv2.GaussianBlur(target_left_eyebrow_img_gray, (3, 3), 0)##高斯过滤
        target_right_eyebrow_img_gray=cv2.GaussianBlur(target_right_eyebrow_img_gray, (3, 3), 0)
        target_left_eyebrow_img_gray=cv2.erode(target_left_eyebrow_img_gray, None, iterations=3)##腐蚀
        target_right_eyebrow_img_gray=cv2.erode(target_right_eyebrow_img_gray, None, iterations=3) 
        target_left_eyebrow_img_gray=cv2.dilate(target_left_eyebrow_img_gray, None, iterations=5)##膨胀 
        target_right_eyebrow_img_gray=cv2.dilate(target_right_eyebrow_img_gray, None, iterations=5)

        target_left_eyebrow_img_gray=cv2.bitwise_not(target_left_eyebrow_img_gray)##取反---模板mask
        target_right_eyebrow_img_gray=cv2.bitwise_not(target_right_eyebrow_img_gray)
        # cv2.imshow('111',target_right_eyebrow_img_gray)

        guide_img_left = cv2.bitwise_and(self.blue_left,self.blue_left, mask=target_left_eyebrow_img_gray) ##原始提示图片
        guide_img_right = cv2.bitwise_and(self.blue_right,self.blue_right, mask=target_right_eyebrow_img_gray)  
        if self.resize_ratio>1 :
            guide_img_left = cv2.resize(guide_img_left,None,fx=self.resize_ratio, fy=self.resize_ratio, interpolation = cv2.INTER_CUBIC)##目标人脸变化到用户人脸大小
            guide_img_right = cv2.resize(guide_img_right,None,fx=self.resize_ratio, fy=self.resize_ratio, interpolation = cv2.INTER_CUBIC)
        if self.resize_ratio<1 :
            guide_img_left = cv2.resize(guide_img_left,None,fx=self.resize_ratio, fy=self.resize_ratio, interpolation =cv2.INTER_AREA)
            guide_img_right = cv2.resize(guide_img_right,None,fx=self.resize_ratio, fy=self.resize_ratio, interpolation =cv2.INTER_AREA)
        
        self.t_left_reference=(self.resize_ratio*self.t_left_reference[0],self.resize_ratio*self.t_left_reference[1])
        self.t_right_reference=(self.resize_ratio*self.t_right_reference[0],self.resize_ratio*self.t_right_reference[1])
        left_c_left=self.t_left_reference[0]
        left_c_top=self.t_left_reference[1]
        left_c_right=guide_img_left.shape[1]-self.t_left_reference[0]
        left_c_bottom=guide_img_left.shape[0]-self.t_left_reference[1]
        self.u_left_reference=self.user_keypoints.left_eyebrow()[4]
        w1=int(self.u_left_reference[1] - left_c_top)
        w2=int(self.u_left_reference[1] + left_c_bottom)
        w3=int(self.u_left_reference[0] - left_c_left)
        w4=int(self.u_left_reference[0] + left_c_right)
        roi_left=self.user_face[w1:w2,w3:w4]
        
        hsv=cv2.cvtColor(guide_img_left,cv2.COLOR_BGR2HSV)
        lower_black=np.array([0,0,0])
        upper_black=np.array([255,255,46])
        # color_1=[255,255,255]
        mask_black=cv2.inRange(hsv,lower_black,upper_black)
        guide_img_left[mask_black!=0]=roi_left[mask_black!=0]



        alpha = 0.8
        beta = 1-alpha
        gamma = 0
        guide_left_eyebrow = cv2.addWeighted(roi_left,alpha,guide_img_left,beta,gamma)

        self.guid_img=self.user_face.copy()
        self.guid_img[w1:w2,w3:w4]=guide_left_eyebrow

        # right_c_left=self.t_right_reference[0]
        right_c_top=self.t_right_reference[1]
        right_c_right=guide_img_right.shape[1]-self.t_right_reference[0]
        right_c_bottom=guide_img_right.shape[0]-self.t_right_reference[1]
        self.u_right_reference=self.user_keypoints.right_eyebrow()[0]

        w1=int(self.u_right_reference[1] - right_c_top)
        w2=int(self.u_right_reference[1] + right_c_bottom)
        w3=int(self.u_right_reference[0] )
        w4=int(self.u_right_reference[0] + right_c_right)
        roi_right=self.user_face[w1:w2,w3:w4]    

        hsv=cv2.cvtColor(guide_img_right,cv2.COLOR_BGR2HSV)
        lower_black=np.array([0,0,0])
        upper_black=np.array([255,255,46])
        # color_1=[255,255,255]
        mask_black=cv2.inRange(hsv,lower_black,upper_black)
        guide_img_right[mask_black!=0]=roi_right[mask_black!=0]

        guide_right_eyebrow = cv2.addWeighted(roi_right,alpha,guide_img_right,beta,gamma)
        self.guid_img[w1:w2,w3:w4]=guide_right_eyebrow
        return self.guid_img
        # cv2.imshow('guid',self.guid_img)

     