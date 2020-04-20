import numpy as np 
import cv2
from matplotlib import pyplot as plt



class LK_tracker:
    def __init__(self,bounds=[]):
        self.bounds = []

        self.last_frame = None
        
        # self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)



        
    def apply(self,img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if len(self.bounds) == 0:
            self.get_start_bound(gray)
            
            self.template = gray[self.bounds[1]:self.bounds[3],self.bounds[0]:self.bounds[2]]
            self.template_size = self.template.shape
            
            self.last_frame = gray
            self.shape = img.shape

            self.corners = cv2.goodFeaturesToTrack(self.template,25,0.01,10)
            self.corners = np.squeeze(np.int0(self.corners))
            self.corners[:,0] = self.corners[:,0] + self.bounds[0]
            self.corners[:,1] = self.corners[:,1] + self.bounds[1]
            
            # print(self.corners)

            self.last_grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
            self.last_grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
            
            return self.last_frame



        self.grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
        self.grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)   

        for indx in range(self.corners.shape[0]):
            self.compute_warp(gray,indx)
        
        self.last_grad_x = self.grad_x
        self.last_grad_y = self.grad_y

        self.last_frame = gray
        
        out_image = self.label_corners(gray.copy())

        return out_image

    def compute_warp(self,img, indx):

        window = 10

        pnt = self.corners[indx,:]
        if pnt[0] + window > self.shape[0] or pnt[1] + window > self.shape[1]:
            return

        x1 = np.matrix([[i for i in range(window)] for j in range(window)])
        y1 = np.matrix([[z] * window for j in range(window)])

        Wxp = [np.multiply(x1, self.grad_last_x), np.multiply(x1, self.grad_last_y), np.multiply(y1, self.grad_last_x),np.multiply(y1, self.grad_last_y), self.grad_last_x, self.grad_last_y]
        
        H = [[np.sum(np.multiply(Wxp[i], Wxp[j])) for i in range(6)] for j in range(6)]
        Hinv = np.linalg.pinv(HessianOriginal)

        p = np.zeros((6,1))

        
        k = 0
        bad_itr = 0
        min_cost = -1
        minW = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        W = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        while (k <= 10):
            position = [[W.dot(np.matrix([[x + i], [y + j], [1]], dtype='float')) for i in range(window)] for j in range(window)]
            
            # if not (0 <= (position[0][0])[0, 0] < cols and 0 <= (position[0][0])[1, 0] < rows and 0 <= position[size - 1][0][
            #     0, 0] < cols and 0 <= position[size - 1][0][1, 0] < rows and 0 <= position[0][size - 1][0, 0] < cols and 0 <=
            #     position[0][size - 1][1, 0] < rows and 0 <= position[size - 1][size - 1][0, 0] < cols and 0 <=
            #     position[size - 1][size - 1][1, 0] < rows):
            #     return np.matrix([[-1], [-1]])
            
            
            
            I = np.matrix([[frame[int((position[i][j])[1, 0]), int((position[i][j])[0, 0])] for j in range(size)] for i in range(size)])

            error = np.absolute(np.matrix(I, dtype='int') - np.matrix(T, dtype='int'))

            steepest_error = np.matrix([[np.sum(np.multiply(g, error))] for g in gradOriginalP])
            mean_cost = np.sum(np.absolute(steepest_error))
            deltap = Hessianinv.dot(steepest_error)


            dp = warpInv(deltap)
            inverse_output = np.matrix([[0.1]] * 6)
            val = (1 + p[0, 0]) * (1 + p[3, 0]) - p[1, 0] * p[2, 0]
            inverse_output[0, 0] = (-p[0, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
            inverse_output[1, 0] = (-p[1, 0]) / val
            inverse_output[2, 0] = (-p[2, 0]) / val
            inverse_output[3, 0] = (-p[3, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
            inverse_output[4, 0] = (-p[4, 0] - p[3, 0] * p[4, 0] + p[2, 0] * p[5, 0]) / val
            inverse_output[5, 0] = (-p[5, 0] - p[0, 0] * p[5, 0] + p[1, 0] * p[4, 0]) / val


            p1, p2, p3, p4, p5, p6 = p1 + dp[0, 0] + p1 * dp[0, 0] + p3 * dp[1, 0], p2 + dp[1, 0] + dp[0, 0] * p2 + p4 * dp[1, 0], p3 + dp[2, 0] + p1 * dp[2, 0] + p3 * dp[3, 0], p4 + dp[3, 0] + p2 * dp[2, 0] + p4 * dp[3, 0], p5 + \
                                     dp[4, 0] + p1 * dp[4, 0] + p3 * dp[5, 0], p6 + dp[5, 0] + p2 * dp[4, 0] + p4 * dp[5, 0]
            W = np.matrix([[1+p1,p3,p5], [p2,1+p4,p6]])

            if (min_cost == -1):
                min_cost = mean_cost
            elif (min_cost >= mean_cost):
                min_cost = mean_cost
                bad_itr = 0
                minW = W
            else:
                bad_itr += 1
            if (bad_itr == 3):
                W = minW
                return W.dot(np.matrix([[x], [y], [1.0]]))

            if (np.sum(np.absolute(deltap)) < 0.0006):
                return W.dot(np.matrix([[x], [y], [1.0]]))
        

    def label_corners(self,img):
        temp = img.copy()
        for i in range(self.corners.shape[0]):
            print((self.corners[i,0],self.corners[i,1]))
            temp = cv2.circle(temp,(self.corners[i,0],self.corners[i,1]),3,255,-1)
        return temp


    def bound_callback(self,event,x,y,flags,param):
        # print(event,cv2.EVENT_LBUTTONDBLCLK)
        if event == 4:
            if len(self.bounds) == 0:
                self.bounds = [x,y]
            elif len(self.bounds) == 2:
                self.bounds.extend([x,y])
                print(self.bounds)


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
