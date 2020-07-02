import dlib
import cv2
import sys
import numpy as np
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from imutils import face_utils

class target_face_detection(object):

    def __init__(self,image,predictor_path):
        self.image = image
        self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        self.predictor_path = predictor_path
    
    def __face_detection(self):
        detector = dlib.get_frontal_face_detector()
        rects = detector(self.gray, 1)
        if len(rects)==0 :
            print("warning!--------no face!")
            sys.exit()
        max_face = (0,rects[0],0)
        for (i,rect) in enumerate(rects):
            area = (rect.right() - rect.left()) * (rect.bottom() - rect.top())
            if area > max_face[2]:
                max_face = (i,rect,area)
        self.face = max_face[1]
    
    def get_key_point(self):
        self.__face_detection()
        predictor = dlib.shape_predictor(self.predictor_path)
        self.shape = predictor(self.gray, self.face)  # 标记人脸中的68个landmark点
        
        self.shape_np = face_utils.shape_to_np(self.shape)  # shape转换成68个坐标点矩阵  
        # print(self.shape_np,self.shape_np.shape,self.shape_np[0:3])
        return self.shape_np

    def face_correction(self):
        self.face_temp=dlib.full_object_detections()
        self.face_temp.append(self.shape)
        self.corrected_face = dlib.get_face_chips(self.image, self.face_temp, size=320)
        print(self.corrected_face,type(self.corrected_face))
        cv_rgb_image = np.array(self.corrected_face).astype(np.uint8)# 先转换为numpy数组
        cv_bgr_image = cv2.cvtColor(cv_rgb_image, cv2.COLOR_RGB2BGR)# opencv下颜色空间为bgr，所以从rgb转换为bgr
        # print(cv_bgr_image.shape)
        cv2.imshow("corrected", cv_bgr_image)


    def turn_to_vector_space(self,model_path):
        var_exists = 'self.shape' in locals()
        if not var_exists :
            self.get_key_point()
        self.model_path = model_path
        face_rec_model = dlib.face_recognition_model_v1(self.model_path)
        b, g, r = cv2.split(self.image)
        img2 = cv2.merge([r, g, b])
        face_descriptor = face_rec_model.compute_face_descriptor( img2, self.shape)
        # print(face_descriptor)


class face_key_point_classification(target_face_detection):
    def jaw_line(self):
        var_exists = 'self.shape_np' in locals()
        if not var_exists :
            self.get_key_point()
        return self.shape_np[0:17]
    
    def left_eyebrow(self):
        var_exists = 'self.shape_np' in locals()
        if not var_exists :
            self.get_key_point()
        return self.shape_np[17:22]    
    
    def right_eyebrow(self):
        var_exists = 'self.shape_np' in locals()
        if not var_exists :
            self.get_key_point()
        return self.shape_np[22:27]


    def ridge(self):
        var_exists = 'self.shape_np' in locals()
        if not var_exists :
            self.get_key_point()
        return self.shape_np[27:31]
    def alar_base(self):
        var_exists = 'self.shape_np' in locals()
        if not var_exists :
            self.get_key_point()
        return self.shape_np[31:36]
    def left_eye(self):
        var_exists = 'self.shape_np' in locals()
        if not var_exists :
            self.get_key_point()
        return self.shape_np[36:42]
                       
    def right_eye(self):
        var_exists = 'self.shape_np' in locals()
        if not var_exists :
            self.get_key_point()
        return self.shape_np[42:48]

    def lips_shape(self):
        var_exists = 'self.shape_np' in locals()
        if not var_exists :
            self.get_key_point()
        return self.shape_np[48:60]    

    def mouth_inner_edge(self):
        var_exists = 'self.shape_np' in locals()
        if not var_exists :
            self.get_key_point()
        return self.shape_np[60:68]

    def min_face_area(self):
        x_min_q=self.left_eyebrow()
        x_max_q=self.right_eyebrow()
        width=(max(x_max_q[:,0]))-(min(x_min_q[:,0]))
        y_min_q=self.right_eyebrow()
        y_max_q=self.alar_base()
        height=(max(y_max_q[:,1]))-(min(y_min_q[:,1]))
        a=int(width*height)
        return a

    def face_width(self):
        var_exists = 'self.shape_np' in locals()
        if not var_exists :
            self.get_key_point()        
        n=self.shape_np[16,0] - self.shape_np[0,0]
        return n

    def eye_distance_ratio(self):
        var_exists = 'self.shape_np' in locals()
        if not var_exists :
            self.get_key_point()  
        n = ( self.shape_np[42,0] - self.shape_np[39,0] )/ (float(self.face_width()))
        return n

    def eye_length_ratio(self):
        var_exists = 'self.shape_np' in locals()
        if not var_exists :
            self.get_key_point()  
        m = (self.shape_np[39,0] - self.shape_np[36,0])+(self.shape_np[45,0] - self.shape_np[42,0])
        n = m/2.0/self.face_width()
        return n

    def face_scale_ratio(self):
        var_exists = 'self.shape_np' in locals()
        if not var_exists :
            self.get_key_point()  
        temp1 = (sum(self.shape_np[18:21,1]) + sum(self.shape_np[23:26,1]) )/6.0
        f1= temp1 - self.shape_np[33,1]
        f2= float(self.shape_np[33,1] - self.shape_np[8,1])
        n = f1 / f2
    
    def eyebrow_length_ratio(self):
        var_exists = 'self.shape_np' in locals()
        if not var_exists :
            self.get_key_point()  
        temp1 = self.shape_np[26,0] - self.shape_np[22,0] + self.shape_np[21,0] - self.shape_np[18,0]
        n = temp1 /2.0 / self.face_width()
        return n

    def eyebrow_distance_ratio(self):
        var_exists = 'self.shape_np' in locals()
        if not var_exists :
            self.get_key_point()  
        n = (self.shape_np[22,0]-self.shape_np[21,0])/(float(self.face_width()))
        return n






