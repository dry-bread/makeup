import cv2
from Face_Module.Get_key_point import face_key_point_classification
from Face_Module.Picture_segmentation import segmentation
from Face_Module.Face_extraction import facial_extraction
from Face_Module.Fleck_detection import find_skin_fleck
import numpy as np
import sys
from Face_Module.Eyes_makeup import eyes_make_up_guide
from Face_Module.Eyebrow_makeup import eyebrow_make_up_guide
from Face_Module.Lip_makeup import lip_make_up_guide
import uuid

UPLOAD_FOLDER = 'static/photo'
STATIC_FOLDER = 'static/img'

face_landmarks_path = "D:\\anaconda_5_3_1\\Anaconda_3\\envs\py3.6-face\\Lib\\site-packages\\dlib\\examples\\shape_predictor_68_face_landmarks.dat"
# face_landmarks_path = "D:\\anaconda_5_3_1\\Anaconda_3\\envs\py3.6-face\\Lib\\site-packages\\dlib\\examples\\shape_predictor_68_face_landmarks.dat"


def makeup(faceFile:str, targetFile):
    res = []
    suffix = faceFile.rsplit('.', 1)[1].lower()
    target_suffix = targetFile.rsplit('.', 1)[1].lower()
    print("python------------")
    image = cv2.imread(UPLOAD_FOLDER+'/'+faceFile)
    target = cv2.imread(STATIC_FOLDER+'/'+targetFile)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #######-------
    # cv2.namedWindow("makeup guide", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("makeup guide", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_GUI_NORMAL)
    # cv2.imshow("makeup guide",image)
    # cv2.waitKey(0)
    # cv2.imshow("makeup guide",target)
    # cv2.waitKey(0)
    #####-------------
    f1 = face_key_point_classification(image,face_landmarks_path)
    # #######--------------------------------

    left_eyebrow=f1.left_eyebrow()
    jaw_line= f1.jaw_line()
    min_area=f1.min_face_area()
    fa=facial_extraction(image,jaw_line,min_area)
    face_img,mask_img=fa.facial_extract()
    if type(face_img) == int :
        print("脸部图像不合格！")
        return res
    
    show_img = find_skin_fleck(face_img).find_fleck()
    filename = faceFile+"-"+targetFile + "zhexia." + suffix
    res.append(("遮瑕", filename))
    cv2.imwrite(UPLOAD_FOLDER + "/" + filename, show_img)
    print(UPLOAD_FOLDER + "/" + filename, show_img)

    #######---------------------------------------------------------------
    ft = face_key_point_classification(target,face_landmarks_path)
    ###333#----------------
    eye_m =eyes_make_up_guide(target,ft,image,f1)
    show_img=eye_m.makeup_guide()
    filename = faceFile+"-"+targetFile + "yanying." + suffix
    res.append(("眼影", filename))
    cv2.imwrite(UPLOAD_FOLDER + "/" + filename, show_img)
    print(UPLOAD_FOLDER + "/" + filename, show_img)

    ######3-----------------
    eyebrow_m =eyebrow_make_up_guide(target,ft,image,f1)
    show_img=eyebrow_m.makeup_guide()
    filename = faceFile+"-"+targetFile + "meizhuang." + suffix
    res.append(("眉妆", filename))
    cv2.imwrite(UPLOAD_FOLDER + "/" + filename, show_img)
    print(UPLOAD_FOLDER + "/" + filename, show_img)
    #3###33###3-------------------

    lip_m=lip_make_up_guide(image,f1)
    show_img=lip_m.makeup_guide()
    filename = faceFile+"-"+targetFile + "kouhong." + suffix
    res.append(("口红", filename))
    cv2.imwrite(UPLOAD_FOLDER + "/" + filename, show_img)
    print(UPLOAD_FOLDER + "/" + filename, show_img)

    return res
