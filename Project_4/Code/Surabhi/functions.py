import numpy as np 
import cv2
from matplotlib import pyplot as plt
import os
dirpath = os.getcwd()

class get_frames:
    def __init__(self,video_seq):
        self.frame_num = 1
        
        self.vid = video_seq
        if video_seq == 1:
            self.file_route = '/Car4/img/'
        elif video_seq == 2:
            self.file_route = '/Bolt2/img/'
        else:
            self.file_route = '/DragonBaby/img/'
          
    def get_next_frame(self):
        num = str(self.frame_num)
        num = num.zfill(4)
        read_frame = cv2.imread(dirpath + self.file_route+num+'.jpg')
        self.frame_num = self.frame_num + 1
        return read_frame

    def get_frame(self):
        # num = str(indx)
        # num = num.zfill(4)
        
        # read_frame = cv2.imread(self.file_route+num+'.jpg')

        # return read_frame
        num = str(self.frame_num)
        num = num.zfill(4)
        read_frame = cv2.imread(dirpath + self.file_route+num+'.jpg')
        self.frame_num = self.frame_num + 1
        return cv2.cvtColor(read_frame, cv2.COLOR_BGR2GRAY)

    def crop_im(self,img, bounds):
        temp_x = int(bounds[0])
        temp_y = int(bounds[1])
        temp_h = int(abs(bounds[1] - bounds[3]))
        temp_w = int(abs(bounds[0] - bounds[2]))
        print("{} {} {} {}".format(temp_x, temp_y, temp_h, temp_w))
        return img[temp_y:temp_y+temp_h, temp_x:temp_x+temp_w]

    def get_bounds(self):
        if self.vid == 1: 
            rect = np.asarray([68, 49, 180, 137])
            return rect
        if self.vid == 2: 
            rect = np.asarray([269, 79, 310, 145])
            return rect


class LucasKanade:
    def __init__(self,video_seq,bounds=None,):
        # coordinates of the two click points
        self.bounds = []
        self.vid = video_seq
            
    def apply(self,img):
        if len(self.bounds) == 0:
            self.get_start_bound(img)
            
        I_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return I_x, self.bounds

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

    # Additive Alignment (Affine); Need to solve for dp
    def align(self, T, I, rect, p, dp0=np.zeros(6), threshold=0.001, iterations=50):

        cap = get_frames(self.vid)
        T_rows, T_cols = T.shape
        I_rows, I_cols = I.shape
        dp = dp0

        for i in range (iterations):
            # Forward warp matrix from frame_t to frame_t+1
            W = np.float32([ 
                [1+p[0], p[2], p[4]], 
                [p[1], 1+p[3], p[5]] ])

            # Warp image from frame_t+1 to frame_t and crop it
            I_warped = cv2.warpAffine(I, cv2.invertAffineTransform(W), (I_cols, I_rows))
            I_warped = cap.crop_im(I_warped, rect)
            
            # Image gradients
            dI_x = cv2.Sobel(I_warped, cv2.CV_64F, 1, 0, ksize=3).flatten()
            dI_y = cv2.Sobel(I_warped, cv2.CV_64F, 0, 1, ksize=3).flatten()
            
            # Hessian
            H = np.zeros((6,6))
            for y in range(T_rows):
                for x in range(T_cols):
                    dW = np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]])
                    dI = np.array([dI_x[x*y], dI_y[x*y]]).reshape(1,2)
                    A = np.matmul(dI, dW)
                    H += A.T*A
                    # A = np.vstack(( A, np.matmul(dI, dW).reshape(1,6) ))
                    # print(np.shape(A))
            
            # Steepest descent
            # A = np.sum(A, axis=0).reshape(1,6)
            # Hessian 
            # print(np.shape(A))
            # H = np.matmul(A.T, A)
            # w,v = np.linalg.eig(H)
            # print(w)
            # Error image 
            err_im = (T - I_warped).flatten()
            err_im = np.reshape(err_im, (1, len(err_im)))

            del_p = np.sum(np.matmul(np.linalg.inv(H), np.matmul(A.T, err_im)), axis=1)

            # Test for convergence and exit 
            if np.linalg.norm(del_p) <= threshold: 
                break

            # Update the parameters
            p = p + del_p

        return p
