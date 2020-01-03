import cv2
import numpy as np
from functools import reduce

class facial_extraction(object):
    def __init__(self,img,jaw_line,min_face_are):
        self.image=img
        self.jaw_line=jaw_line
        self.min_face_are=min_face_are
    
    def cr_otsu(self):
        """YCrCb颜色空间的Cr分量+Otsu阈值分割"""
        ycrcb = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCR_CB)
        (y, cr, cb) = cv2.split(ycrcb)
        cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
        _, self.skin_img = cv2.threshold(cr1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.canny_img=cv2.Canny(self.skin_img, 200, 300)
        self.mid_img = cv2.bitwise_and(self.image, self.image, mask=self.skin_img)
    
    
    def facial_extract(self):
        self.cr_otsu() 
        last_point=(self.jaw_line[0,0],self.jaw_line[0,1])
        for index, pt in enumerate(self.jaw_line):
            pt_point = (pt[0],pt[1])
            cv2.line(self.canny_img,last_point,pt_point,(255,255,255),1)
            last_point=pt_point
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        closed = cv2.morphologyEx(self.canny_img, cv2.MORPH_CLOSE, kernel)
        closed = cv2.dilate(closed, None, iterations=6)
        closed = cv2.erode(closed, None, iterations=6)

        contours,__ = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE )

        area = map(lambda i: cv2.contourArea(i), contours)
        try:
            min_idx = reduce(lambda x,y: x if x[1] < y[1] else y,filter(lambda i: i[1] >= self.min_face_are, enumerate(area)))[0]
        except:
            print("请重新输入图像！")
            return -1
        else:
            cv2.fillConvexPoly(self.canny_img, contours[min_idx], 255)
            closed = cv2.erode(self.canny_img, None, iterations=6)
            final_mask = cv2.dilate(closed, None, iterations=5) 
            self.final_face = cv2.bitwise_and(self.image, self.image, mask=final_mask)  
            self.mask_img=cv2.bitwise_not(final_mask) 
            # cv2.imshow('final face',self.final_face)
            # cv2.imshow('mask img',self.mask_img)            
            return self.final_face,self.mask_img
        







    



