import numpy as np 
import cv2
from matplotlib import pyplot as plt



class LK_tracker:
    def __init__(self,bounds=None):
        self.bounds = []
            
        
    def apply(self,img):
        if len(self.bounds) == 0:
            self.get_start_bound(img)
            
        out_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return out_image


    def bound_callback(self,event,x,y,flags,param):
        # print(event,cv2.EVENT_LBUTTONDBLCLK)
        if event == 4:
            if len(self.bounds) == 0:
                self.bounds = [x,y]
            elif len(self.bounds) == 2:
                self.bounds.extend([x,y])

    def get_start_bound(self,img):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',self.bound_callback)
        cv2.imshow('image',img)

        while len(self.bounds) < 4:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

        

class get_frames:
    def __init__(self,video_seq):
        self.frame_num = 1
        
        self.vid = video_seq
        if video_seq == 1:
            self.file_route = 'Car4/img/'
        elif video_seq == 2:
            self.file_route = 'Bolt/img/'
        else:
            self.file_route = 'DragonBaby/img/'

            
    def get_next_frame(self):
        num = str(self.frame_num)
        num = num.zfill(4)
        
        read_frame = cv2.imread(self.file_route+num+'.jpg')
        self.frame_num = self.frame_num + 1
        return read_frame

    def get_frame(self,indx):
        num = str(indx)
        num = num.zfill(4)
        
        read_frame = cv2.imread(self.file_route+num+'.jpg')

        return read_frame
