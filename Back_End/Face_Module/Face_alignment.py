# encoding:utf-8

import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import leastsq


class Key_points_alignment(object):
    def __init__(self,face_img,left_eye_np,right_eye_np,ridge_np):
        self.face_img=face_img
        self.left_eye_np=left_eye_np
        self.right_eye_np=right_eye_np
        self.ridge_np=ridge_np
        

    def alignment(self):
        left_eye_center_x=sum(self.left_eye_np[:,0])
        left_eye_center_y=sum(self.left_eye_np[:,1])
        right_eye_center_x=sum(self.right_eye_np[:,0])
        right_eye_center_y=sum(self.right_eye_np[:,1])
        eye_center = ( (left_eye_center_x + right_eye_center_x)/2.0 ,(left_eye_center_y + right_eye_center_y)/2.0 )
        
        eye_dy=right_eye_center_y-left_eye_center_y
        eye_dx=right_eye_center_x-left_eye_center_x
        eye_angle=math.atan2(eye_dy,eye_dx) * 180. / math.pi # 计算角度

        ridge_angle=self.get_ridge_angle()

        final_angle = (eye_angle + ridge_angle)/2.0

        RotateMatrix = cv2.getRotationMatrix2D(eye_center, final_angle, scale=1) # 计算仿射矩阵
        RotImg = cv2.warpAffine(self.face_img, RotateMatrix,self.face_img.shape ) # 进行放射变换，即旋转
        return RotImg






    
    def fun(self,p, x):
       k, b = p  #从参数p获得拟合的参数
       return k*x+b 
    def err(self.p, x, y):
       """
       定义误差函数
       """
       return self.fun(p,x) -y
    def get_ridge_angle(self):
        p0 = self.ridge_np[0]
        x = self.ridge_np[:,0]
        y = self.ridge_np[:,1]
        xishu = leastsq(self.err, p0, args=(x,y))
        r = math.atan2(xishu[0][0],1) * 180. / math.pi
        return r










def face_alignment(faces):

    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # 用来预测关键点
    faces_aligned = []
    for face in faces:
        rec = dlib.rectangle(0,0,face.shape[0],face.shape[1])
        shape = predictor(np.uint8(face),rec) # 注意输入的必须是uint8类型
        order = [36,45,30,48,54] # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序，这个在网上可以找
        for j in order:
            x = shape.part(j).x
            y = shape.part(j).y
            cv2.circle(face, (x, y), 2, (0, 0, 255), -1)

        eye_center =((shape.part(36).x + shape.part(45).x) * 1./2, # 计算两眼的中心坐标
                      (shape.part(36).y + shape.part(45).y) * 1./2)
        dx = (shape.part(45).x - shape.part(36).x) # note: right - right
        dy = (shape.part(45).y - shape.part(36).y)

        angle = math.atan2(dy,dx) * 180. / math.pi # 计算角度
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1) # 计算仿射矩阵
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1])) # 进行放射变换，即旋转
        faces_aligned.append(RotImg)
    return faces_aligned

def demo():


    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    src_faces = []
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = rect_to_bb(rect)
        detect_face = im_raw[y:y+h,x:x+w]
        src_faces.append(detect_face)
        cv2.rectangle(im_raw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(im_raw, "Face: {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    faces_aligned = face_alignment(src_faces)

    cv2.imshow("src", im_raw)
    i = 0
    for face in faces_aligned:
        cv2.imshow("det_{}".format(i), face)
        i = i + 1
    cv2.waitKey(0)

if __name__ == "__main__":

    demo()

