import numpy as np 
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D  



class color_data:
    def __init__(self):
        if False:
            self.train1 = np.load('color_data/training_buoy1_data.npy')
            self.train2 = np.load('color_data/training_buoy2_data.npy')
            self.train3 = np.load('color_data/training_buoy3_data.npy')
            self.test1 = np.load('color_data/testing_buoy1_data.npy')
            self.test2 = np.load('color_data/testing_buoy2_data.npy')
            self.test3 = np.load('color_data/testing_buoy3_data.npy')
        else:
            self.train1 = np.load('color_data/training_buoy1_data_HSV.npy')
            self.train2 = np.load('color_data/training_buoy2_data_HSV.npy')
            self.train3 = np.load('color_data/training_buoy3_data_HSV.npy')
            self.test1 = np.load('color_data/testing_buoy1_data_HSV.npy')
            self.test2 = np.load('color_data/testing_buoy2_data_HSV.npy')
            self.test3 = np.load('color_data/testing_buoy3_data_HSV.npy')

    
    def plot_data(self,color_data):

        if color_data is None:
            B = self.test3[:,0]
            G = self.test3[:,1]
            R = self.test3[:,2]
        else:
            B = color_data[:,0]
            G = color_data[:,1]
            R = color_data[:,2]
        
        # Make a 3D ScaTTER Of RGB values
        fig = plt.figure()
        ax = Axes3D(fig)
        
        col = np.vstack([R, G, B])/255.0
        ax.scatter(R, G, B, c = np.transpose(col))

        ax.set_xlabel('B')
        ax.set_ylabel('G')
        ax.set_zlabel('R')

        plt.show()


class color_mask:
    def __init__(self):
        self.cap = cv2.VideoCapture('buoy_mask.avi')
        self.buoy_pnts = np.load('buoy_points.npy')
        self.thres = [127,255]
        self.kernel3 = np.ones((3,3),np.uint8)
        self.kernel5 = np.ones((9,9),np.uint8)
        self.wnd = 20
        self.thres = [80,80,40]

    def get_mask(self,img,frame_num):
        if frame_num > 100:
            self.wnd = 50
        else:
            self.wnd = 20

        buoy_pnt_1 = self.buoy_pnts[frame_num,0:2]
        buoy_pnt_2 = self.buoy_pnts[frame_num,2:4]
        buoy_pnt_3 = self.buoy_pnts[frame_num,4:6]
        bds1 = [max(0,buoy_pnt_1[1]-self.wnd),min(480,buoy_pnt_1[1]+self.wnd)+1,max(0,buoy_pnt_1[0]-self.wnd),min(640,buoy_pnt_1[0]+self.wnd)+1]
        bds2 = [max(0,buoy_pnt_2[1]-self.wnd),min(480,buoy_pnt_2[1]+self.wnd)+1,max(0,buoy_pnt_2[0]-self.wnd),min(640,buoy_pnt_2[0]+self.wnd)+1]
        bds3 = [max(0,buoy_pnt_3[1]-self.wnd),min(480,buoy_pnt_3[1]+self.wnd)+1,max(0,buoy_pnt_3[0]-self.wnd),min(640,buoy_pnt_3[0]+self.wnd)+1]
        
        out_image = np.zeros((480, 640, 3), dtype=np.uint8)



        buoy1_img = img[bds1[0]:bds1[1],bds1[2]:bds1[3]]
        color1u = np.array([img[buoy_pnt_1[1],buoy_pnt_1[0],0]+self.thres[0],img[buoy_pnt_1[1],buoy_pnt_1[0],1]+self.thres[0],img[buoy_pnt_1[1],buoy_pnt_1[0],2]+self.thres[0]])
        color1b = np.array([img[buoy_pnt_1[1],buoy_pnt_1[0],0]-self.thres[0],img[buoy_pnt_1[1],buoy_pnt_1[0],1]-self.thres[0],img[buoy_pnt_1[1],buoy_pnt_1[0],2]-self.thres[0]])
        mask1 = cv2.inRange(buoy1_img, np.clip(color1b,0,255), np.clip(color1u,0,255))
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, self.kernel3)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, self.kernel5)
        out_image[bds1[0]:bds1[1],bds1[2]:bds1[3],0] = mask1


        if (buoy_pnt_2 != [0,0]).all():
            buoy2_img = img[bds2[0]:bds2[1],bds2[2]:bds2[3]]
            color2u = np.array([img[buoy_pnt_2[1],buoy_pnt_2[0],0]+self.thres[1],img[buoy_pnt_2[1],buoy_pnt_2[0],1]+self.thres[1],img[buoy_pnt_2[1],buoy_pnt_2[0],2]+self.thres[1]])
            color2b = np.array([img[buoy_pnt_2[1],buoy_pnt_2[0],0]-self.thres[1],img[buoy_pnt_2[1],buoy_pnt_2[0],1]-self.thres[1],img[buoy_pnt_2[1],buoy_pnt_2[0],2]-self.thres[1]])
            mask2 = cv2.inRange(buoy2_img, np.clip(color2b,0,255), np.clip(color2u,0,255))
            out_image[bds2[0]:bds2[1],bds2[2]:bds2[3],1] = mask2
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, self.kernel3)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, self.kernel5)

        
        if (buoy_pnt_3 != [0,0]).all():
            buoy3_img = img[bds3[0]:bds3[1],bds3[2]:bds3[3]]
            color3u = np.array([img[buoy_pnt_3[1],buoy_pnt_3[0],0]+self.thres[2],img[buoy_pnt_3[1],buoy_pnt_3[0],1]+self.thres[2],img[buoy_pnt_3[1],buoy_pnt_3[0],2]+self.thres[2]])
            color3b = np.array([img[buoy_pnt_3[1],buoy_pnt_3[0],0]-self.thres[2],img[buoy_pnt_3[1],buoy_pnt_3[0],1]-self.thres[2],img[buoy_pnt_3[1],buoy_pnt_3[0],2]-self.thres[2]])
            mask3 = cv2.inRange(buoy3_img, np.clip(color3b,0,255), np.clip(color3u,0,255))
            out_image[bds3[0]:bds3[1],bds3[2]:bds3[3],2] = mask3
            mask3 = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, self.kernel5)
            mask3 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, self.kernel5)

        return out_image

        
        # color_seg1 = buoy1_img.reshape(((bds1[1]-bds1[0])*(bds1[3]-bds1[2]),3))
        # color_seg2 = buoy2_img.reshape(((bds2[1]-bds2[0])*(bds2[3]-bds2[2]),3))
        # color_seg3 = buoy3_img.reshape(((bds3[1]-bds3[0])*(bds3[3]-bds3[2]),3))
        
        # mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # return None

        # mean = [240,253,237]
        # covariance = [30,10,10]
        
        # Bp = multivariate_normal.pdf(data_1,mean[0],covariance[0],allow_singular=True)
        # Gp = multivariate_normal.pdf(data_2,mean[1],covariance[1],allow_singular=True)
        # Rp = multivariate_normal.pdf(data_3,mean[2],covariance[2],allow_singular=True)

        # fig = plt.figure()
        # plt.hist(Rp, bins=100, range=(0.0, max(Rp)), fc='r', ec='r')
        # plt.show()
                
        # thresholds = [.06, .1, .1]

        # Bp = np.reshape(Bp, (480,640))
        # Gp = np.reshape(Gp, (480,640))
        # Rp = np.reshape(Rp, (480,640))        
        
        # out_image[Bp > thresholds[0]] = (255,0,0)
        # out_image[Gp > thresholds[1]] = (0,255,0)
        # out_image[Rp > thresholds[2]] = (0,0,255)

    def get_mask_old(self):
        ret, frame_mask = self.cap.read()

        if frame_mask is None:
            return None

        ret,thres = cv2.threshold(frame_mask,self.thres[0],self.thres[1],cv2.THRESH_BINARY)
        (mask1, mask2, mask3) = cv2.split(thres)

        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, self.kernel3)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, self.kernel3)
        mask3 = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, self.kernel3)

        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, self.kernel5)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, self.kernel5)
        mask3 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, self.kernel5)


        return cv2.merge([mask1,mask2,mask3])

    def get_color_arrays(self,img,masks):

        (mask1, mask2, mask3) = cv2.split(masks)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        color_seg1 = img.copy()
        color_seg2 = img.copy()
        color_seg3 = img.copy()
        color_seg1[mask1 == 0] = (0,0,0)
        color_seg2[mask2 == 0] = (0,0,0)
        color_seg3[mask3 == 0] = (0,0,0)
        
        color_seg1 = color_seg1.reshape((480*640,3))
        color_seg2 = color_seg2.reshape((480*640,3))
        color_seg3 = color_seg3.reshape((480*640,3))
        
        color_seg1 = color_seg1[~np.all(color_seg1 == 0, axis=1)]
        color_seg2 = color_seg2[~np.all(color_seg2 == 0, axis=1)]
        color_seg3 = color_seg3[~np.all(color_seg3 == 0, axis=1)]
        
        return color_seg1, color_seg2, color_seg3

    def get_all_arrays(self,img,masks):

        (mask1, mask2, mask3) = cv2.split(masks)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        color_seg1 = img.copy()
        color_seg2 = img.copy()
        color_seg3 = img.copy()
        color_segR = img.copy()
        
        color_seg1[mask1 == 0] = (0,0,0)
        color_seg2[mask2 == 0] = (0,0,0)
        color_seg3[mask3 == 0] = (0,0,0)

        color_segR[mask1 != 0] = (0,0,0)
        color_segR[mask2 != 0] = (0,0,0)
        color_segR[mask3 != 0] = (0,0,0)
        
        color_seg1 = color_seg1.reshape((480*640,3))
        color_seg2 = color_seg2.reshape((480*640,3))
        color_seg3 = color_seg3.reshape((480*640,3))
        color_segR = color_segR.reshape((480*640,3))
        
        color_seg1 = color_seg1[~np.all(color_seg1 == 0, axis=1)]
        color_seg2 = color_seg2[~np.all(color_seg2 == 0, axis=1)]
        color_seg3 = color_seg3[~np.all(color_seg3 == 0, axis=1)]
        color_segR = color_segR[~np.all(color_segR == 0, axis=1)]

        return color_seg1, color_seg2, color_seg3, color_segR 


class GMM:
    def __init__(self,GMM_file=None):

        if GMM_file is not None:
            gmm_data = np.load(GMM_file)

            self.k = gmm_data.shape[0]

            self.weights = gmm_data[:,0]
            self.means = gmm_data[:,1]
            self.covariances = gmm_data[:,2]
            self.threshold = gmm_data[:,3]
        else:
            self.k = 0

            self.weights = None
            self.means = None
            self.covariances = None
            self.threshold = None


    def apply_gmm(self,img):

        # if self._color:
            # img_s = cv2.cvtColor(img, self._color)
        
        y = 480
        x = 640

        data = np.reshape(img, (x*y, 3))

        prob = np.zeros((data.shape[0], self.k))

        # Compute Probabilty
        for i in range(self.k):
            normal = multivariate_normal.pdf(data,self.means[i],self.covariances[i],allow_singular=True)
            pd = self._weights[i] * normal
            prob[:, i] = pd

        # Reshape into image format
        loglikes = np.sum(prob, axis=1)
        prob_img = np.reshape(loglikes, (y, x))

        # Find points above threshold
        out_image = np.zeros((y, x), dtype=np.uint8)
        out_image[loglike_img > self.threshold] = 255
        # out_image[loglike_img > self.median*self.threshold] = 255        
        
        return out_image


