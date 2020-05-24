import cv2

class segmentation(object):
    def __init__(self,image,points_np):
        self.image=image
        self.points_np=points_np
    
    def get_coordinates(self):
        self.left=min(self.points_np[:,0])
        self.right=max(self.points_np[:,0])
        self.top=min(self.points_np[:,1])
        self.bottom=max(self.points_np[:,1])
        # print(left,top,right,bottom)
        coordibates=(self.left,self.top,self.right,self.bottom)
        return coordibates
    def segment_picture(self):
        var_exists = 'self.coordibates' in locals()
        if not var_exists :
            self.get_coordinates()
        self.picture = self.image[self.top:self.bottom,self.left:self.right]
        return self.picture
               


        
        


