import cv2
import dlib
import sys


image1 = cv2.imread("D:\\document\\makeup\\faces\\suyan1.jpg")

def Fcae_detect(imagePath):
    image=cv2.imread(imagePath)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 1)
    if len(rects)==0 :
        return (0,0,0,0)
    max_face = (0,rects[0],0)
    for (i,rect) in enumerate(rects):
        area = (rect.right() - rect.left()) * (rect.bottom() - rect.top())
        if area > max_face[2]:
            max_face = (i,rect,area)
    face = (max_face[1].left(),max_face[1].top(),max_face[1].right(),max_face[1].bottom())
    return face

if __name__ == "__main__":
    m=Fcae_detect(image1)
    print(m,type(m))






