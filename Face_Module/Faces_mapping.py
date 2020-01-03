from Get_key_point import face_key_point_classification

class map_to_user_face(face_key_point_classification):

    def __init__(self,target_face,target_keypoints,user_face,user_keypoints):
        self.target_face=target_face
        self.target_keypoints=target_keypoints
        self.user_face=user_face
        self.user_keypoints=user_keypoints
        self.target_jaw_line=self.target_keypoints.jaw_line()
        self.user_jaw_line=self.user_keypoints.jaw_line()

    def zoom_ratio(self):
        target_face_width = max(self.target_jaw_line[:,0]) - min(self.target_jaw_line[:,0])
        user_face_width = max(self.user_jaw_line[:,0]) - min(self.user_jaw_line[:,0])
        self.ratio = user_face_width / float( target_face_width )
        return self.ratio
        
    
    def resize_target_face(self):
        sized_target_img=cv2.resize(self.target_face,None,fx=self.ratio,fy=self.ratio)
        



