import cv2
from Get_key_point import face_key_point_classification
from Picture_segmentation import segmentation
from  Face_extraction import facial_extraction
from Fleck_detection import find_skin_fleck
import numpy as np
import sys
from Eyes_makeup import eyes_make_up_guide
from Eyebrow_makeup import eyebrow_make_up_guide
from Lip_makeup import lip_make_up_guide

print("python------------")
image = cv2.imread("D:\\document\\makeup\\faces\\suyan1.jpg")
target = cv2.imread("D:\\document\\makeup\\faces\\huazhuang3.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#######-------
cv2.namedWindow("makeup guide", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("makeup guide", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_GUI_NORMAL)
cv2.imshow("makeup guide",image)
cv2.waitKey(0)
cv2.imshow("makeup guide",target)
cv2.waitKey(0)
#####-------------
f1 = face_key_point_classification(image,"D:\\anaconda_5_3_1\\Anaconda_3\\envs\py3.6-face\\Lib\\site-packages\\dlib\\examples\\shape_predictor_68_face_landmarks.dat")
# #######--------------------------------

left_eyebrow=f1.left_eyebrow()
jaw_line= f1.jaw_line()
min_area=f1.min_face_area()
fa=facial_extraction(image,jaw_line,min_area)
face_img,mask_img=fa.facial_extract()
if type(face_img) == int :
    print("脸部图像不合格！")
    sys.exit()
show_img=find_skin_fleck(face_img).find_fleck()
cv2.imshow("makeup guide",show_img)
cv2.waitKey(0)
#######---------------------------------------------------------------
ft = face_key_point_classification(target,"D:\\anaconda_5_3_1\\Anaconda_3\\envs\py3.6-face\\Lib\\site-packages\\dlib\\examples\\shape_predictor_68_face_landmarks.dat")
###333#----------------
eye_m =eyes_make_up_guide(target,ft,image,f1)
show_img=eye_m.makeup_guide()
cv2.imshow("makeup guide",show_img)
cv2.waitKey(0)
######3-----------------
eyebrow_m =eyebrow_make_up_guide(target,ft,image,f1)
show_img=eyebrow_m.makeup_guide()
cv2.imshow("makeup guide",show_img)
cv2.waitKey(0)
#3###33###3-------------------

lip_m=lip_make_up_guide(image,f1)
show_img=lip_m.makeup_guide()
cv2.imshow("makeup guide",show_img)
cv2.waitKey(0)

######33#----------
cv2.destroyAllWindows()