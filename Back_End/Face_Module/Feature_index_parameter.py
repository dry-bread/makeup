from Get_key_point import face_key_point_classification


class Index_parameter(face_key_point_classification):
    def face_width(self):
        n=self.shape_np[16,0] - self.shape_np[0,0]
        return n

    def eye_distance_ratio(self):
        n = ( self.shape_np[42,0] - self.shape_np[39,0] )/ (float(self.face_width()))
        return n

    def eye_length_ratio(self):
        m = (self.shape_np[39,0] - self.shape_np[36,0])+(self.shape_np[45,0] - self.shape_np[42,0])
        n = m/2.0/self.face_width()
        return n

    def face_scale_ratio(self):
        temp1 = (sum(self.shape_np[18:20,1]) + sum(self.shape_np[23:25,1]) )/6.0
        f1= temp1 - self.shape_np[33,1]
        f2= float(self.shape_np[33,1] - self.shape_np[8,1])
        n = f1 / f2
    
    def eyebrow_length_ratio(self):
        temp1 = self.shape_np[26,0] - self.shape_np[22,0] + self.shape_np[21,0] - self.shape_np[18,0]
        n = temp1 /2.0 / self.face_width()
        return n

    def eyebrow_distance_ratio(self):
        n = (self.shape_np[22,0]-self.shape_np[21,0])/(float(self.face_width()))
        return n

    



