from Face_Module.Get_key_point import face_key_point_classification
import numpy as np
import cv2
import gc


class eyes_make_up_guide(object):
    def __init__(self,target_face,target_keypoints:face_key_point_classification,user_face,user_keypoints:face_key_point_classification):
        self.target_face=target_face
        self.target_keypoints=target_keypoints
        self.user_face=user_face
        self.user_keypoints=user_keypoints
        self.blue=cv2.imread("D:\\document\\makeup\\faces\\blue1.jpg")
        self.resize_ratio=user_keypoints.face_width() / target_keypoints.face_width()

    def t_left_eye_pic(self):
        left= 0.58*min(self.target_keypoints.jaw_line()[:,0]) + 0.42*(min(self.target_keypoints.left_eyebrow()[:,0]))
        top = ( min(self.target_keypoints.left_eyebrow()[:,1]) + min(self.target_keypoints.left_eye()[:,1])) / 2
        right = ( max(self.target_keypoints.left_eye()[:,0]) + min(self.target_keypoints.ridge()[:,0]) ) /2
        bottom = max(self.target_keypoints.left_eye()[:,1]) + top - min(self.target_keypoints.left_eyebrow()[:,1])
        self.target_left_eye_img = self.target_face[int(top):int(bottom+1),int(left):int(right+1)]
        self.target_left_eye_points= list(map( lambda i: np.array([i[0]-left,i[1]-top]),self.target_keypoints.left_eye() ))
        self.target_left_eye_points=np.array(self.target_left_eye_points)
        # self.target_left_eye_points= np.array(map( lambda i: np.array([i[0]-left,i[1]-top]),self.target_keypoints.left_eye() ))
        temp1=sum(self.target_left_eye_points[:,0])
        temp2=len(self.target_left_eye_points[:,0])
        left_center_x= temp1/ temp2
        temp1=sum(self.target_left_eye_points[:,1])
        temp2=len(self.target_left_eye_points[:,1])
        left_center_y= temp1/ temp2
        self.t_left_center=(int(left_center_x),int(left_center_y))
        # print(self.t_left_center)


        #im_with_keypoints = cv2.drawKeypoints(self.target_left_eye_img, self.target_left_eye_points, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    def u_left_eye_pic(self):
        temp1=sum(self.user_keypoints.left_eye()[:,0])
        temp2=len(self.user_keypoints.left_eye()[:,0])
        left_center_ux= temp1/temp2 
        temp1=sum(self.user_keypoints.left_eye()[:,1])
        temp2=len(self.user_keypoints.left_eye()[:,1])
        left_center_uy=temp1/temp2
        self.u_left_center=(int(left_center_ux),int(left_center_uy))
        

    def t_right_eye_pic(self):
        right= 0.58*max(self.target_keypoints.jaw_line()[:,0]) + 0.42*(max(self.target_keypoints.right_eye()[:,0]))
        top = ( min(self.target_keypoints.right_eyebrow()[:,1]) + min(self.target_keypoints.right_eye()[:,1])) / 2
        left = ( min(self.target_keypoints.right_eye()[:,0]) + max(self.target_keypoints.ridge()[:,0]) ) /2
        bottom = max(self.target_keypoints.right_eye()[:,1]) + top - min(self.target_keypoints.right_eyebrow()[:,1])
        self.target_right_eye_img = self.target_face[int(top):int(bottom+1),int(left):int(right+1)]
        self.target_right_eye_points= list(map( lambda i: np.array([i[0]-left,i[1]-top]),self.target_keypoints.right_eye() ))
        self.target_right_eye_points =np.array(self.target_right_eye_points)
        temp1=sum(self.target_right_eye_points[:,0])
        temp2=len(self.target_right_eye_points[:,0])
        right_center_x= temp1/ temp2
        temp1=sum(self.target_right_eye_points[:,1])
        temp2=len(self.target_right_eye_points[:,1])
        right_center_y= temp1/ temp2
        self.t_right_center=(int(right_center_x),int(right_center_y))

    def u_right_eye_pic(self):
        right_center_ux=sum(self.user_keypoints.right_eye()[:,0]) / len(self.user_keypoints.right_eye())
        right_center_uy=sum(self.user_keypoints.right_eye()[:,1]) / len(self.user_keypoints.right_eye())
        self.u_right_center=(int(right_center_ux),int(right_center_uy))
    
        

    def makeup_guide(self):
        cv2.fillConvexPoly(self.target_face, self.target_keypoints.left_eye(), (255,2555,255))
        cv2.fillConvexPoly(self.target_face, self.target_keypoints.right_eye(),(255,2555,255))        
        self.t_left_eye_pic()
        self.t_right_eye_pic()
        self.u_left_eye_pic()
        self.u_right_eye_pic()

        self.blue_left = cv2.resize(self.blue,(self.target_left_eye_img.shape[1],self.target_left_eye_img.shape[0]) , interpolation = cv2.INTER_AREA)
        self.blue_right = cv2.resize(self.blue,(self.target_right_eye_img.shape[1],self.target_right_eye_img.shape[0]) , interpolation = cv2.INTER_AREA)
   

        target_left_eye_img_gray =cv2.cvtColor(self.target_left_eye_img,cv2.COLOR_BGR2GRAY)
        target_right_eye_img_gray =cv2.cvtColor(self.target_right_eye_img,cv2.COLOR_BGR2GRAY)

        # target_left_eye_img_gray = cv2.equalizeHist(target_left_eye_img_gray)
        # target_right_eye_img_gray = cv2.equalizeHist(target_right_eye_img_gray)


        _,target_left_eye_img_gray=cv2.threshold(target_left_eye_img_gray,110,255,cv2.THRESH_BINARY)
        _,target_right_eye_img_gray=cv2.threshold(target_right_eye_img_gray,110,255,cv2.THRESH_BINARY)
        target_left_eye_img_gray=cv2.GaussianBlur(target_left_eye_img_gray, (3, 3), 0)
        target_right_eye_img_gray=cv2.GaussianBlur(target_right_eye_img_gray, (3, 3), 0)
        target_left_eye_img_gray=cv2.erode(target_left_eye_img_gray, None, iterations=3)
        target_right_eye_img_gray=cv2.erode(target_right_eye_img_gray, None, iterations=3) 
        target_left_eye_img_gray=cv2.dilate(target_left_eye_img_gray, None, iterations=3)   
        target_right_eye_img_gray=cv2.dilate(target_right_eye_img_gray, None, iterations=3)

        target_left_eye_img_gray=cv2.bitwise_not(target_left_eye_img_gray)
        target_right_eye_img_gray=cv2.bitwise_not(target_right_eye_img_gray)
        

        guide_img_left = cv2.bitwise_and(self.blue_left,self.blue_left, mask=target_left_eye_img_gray) 
        guide_img_right = cv2.bitwise_and(self.blue_right,self.blue_right, mask=target_right_eye_img_gray)  
        if self.resize_ratio>1 :
            guide_img_left = cv2.resize(guide_img_left,None,fx=self.resize_ratio, fy=self.resize_ratio, interpolation = cv2.INTER_CUBIC)
            guide_img_right = cv2.resize(guide_img_right,None,fx=self.resize_ratio, fy=self.resize_ratio, interpolation = cv2.INTER_CUBIC)
        if self.resize_ratio<1 :
            guide_img_left = cv2.resize(guide_img_left,None,fx=self.resize_ratio, fy=self.resize_ratio, interpolation =cv2.INTER_AREA)
            guide_img_right = cv2.resize(guide_img_right,None,fx=self.resize_ratio, fy=self.resize_ratio, interpolation =cv2.INTER_AREA)
        self.t_left_center=(self.resize_ratio*self.t_left_center[0],self.resize_ratio*self.t_left_center[1])
        self.t_right_center=(self.resize_ratio*self.t_right_center[0],self.resize_ratio*self.t_right_center[1])
        left_c_left=self.t_left_center[0]
        left_c_top=self.t_left_center[1]
        left_c_right=guide_img_left.shape[1]-self.t_left_center[0]
        left_c_bottom=guide_img_left.shape[0]-self.t_left_center[1]
        w1=int(self.u_left_center[1] - left_c_top)
        w2=int(self.u_left_center[1] + left_c_bottom)
        w3=int(self.u_left_center[0] - left_c_left)
        w4=int(self.u_left_center[0] + left_c_right)

        
        roi_left=self.user_face[w1:w2,w3:w4]

        hsv=cv2.cvtColor(guide_img_left,cv2.COLOR_BGR2HSV)
        lower_black=np.array([0,0,0])
        upper_black=np.array([255,255,46])
        # color_1=[255,255,255]
        mask_black=cv2.inRange(hsv,lower_black,upper_black)
        guide_img_left[mask_black!=0]=roi_left[mask_black!=0]
        
        alpha = 0.7
        beta = 1-alpha
        gamma = 0
        guide_left_eye = cv2.addWeighted(roi_left,alpha,guide_img_left,beta,gamma)

        self.guid_img=self.user_face.copy()
        self.guid_img[w1:w2,w3:w4]=guide_left_eye



        right_c_left=self.t_right_center[0]
        right_c_top=self.t_right_center[1]
        right_c_right=guide_img_right.shape[1]-self.t_right_center[0]
        right_c_bottom=guide_img_right.shape[0]-self.t_right_center[1]
        w1=int(self.u_right_center[1] - right_c_top)
        w2=int(self.u_right_center[1] + right_c_bottom)
        w3=int(self.u_right_center[0] - right_c_left)
        w4=int(self.u_right_center[0] + right_c_right)
        roi_right=self.user_face[w1:w2,w3:w4]        
        # roi_right=self.user_face[(self.u_right_center[1] - right_c_top):(self.u_right_center[0] + right_c_bottom),(self.u_right_center[0] - right_c_left):(self.u_right_center[0] + right_c_right)]

        guide_right_eye = cv2.addWeighted(roi_right,alpha,guide_img_right,beta,gamma)
        hsv=cv2.cvtColor(guide_img_right,cv2.COLOR_BGR2HSV)
        lower_black=np.array([0,0,0])
        upper_black=np.array([255,255,46])
        # color_1=[255,255,255]
        mask_black=cv2.inRange(hsv,lower_black,upper_black)
        guide_img_right[mask_black!=0]=roi_right[mask_black!=0]

        guide_right_eye = cv2.addWeighted(roi_right,alpha,guide_img_right,beta,gamma)
        self.guid_img[w1:w2,w3:w4]=guide_right_eye
        return self.guid_img
        # cv2.imshow('guid',self.guid_img)


        # cv2.imshow("aaa",guide_left_eye)
        # cv2.imshow("bbb",guide_right_eye)        






    

    
