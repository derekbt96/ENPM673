import numpy as np 
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D  
from scipy.stats import multivariate_normal


class color_data:
    def __init__(self):
        if False:
            self.train1 = np.load('color_data/training_buoy1_data.npy')
        else:
            self.train1 = np.load('color_data/training_buoy1_data_HSV.npy')
            
        if False:
            self.train2 = np.load('color_data/training_buoy2_data.npy')
        else:
            self.train2 = np.load('color_data/training_buoy2_data_HSV.npy')
            
        if False:
            self.train3 = np.load('color_data/training_buoy3_data.npy')
        else:
            self.train3 = np.load('color_data/training_buoy3_data_HSV.npy')
            

    
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
        self.kernel5 = np.ones((5,5),np.uint8)
        self.wnd = 18
        self.thres = [80,80,40]

    def get_mask(self,img,frame_num):
        if frame_num > 100:
            self.wnd = 50
        else:
            self.wnd = 18

        buoy_pnt_1 = self.buoy_pnts[frame_num,0:2]
        buoy_pnt_2 = self.buoy_pnts[frame_num,2:4]
        buoy_pnt_3 = self.buoy_pnts[frame_num,4:6]
        bds1 = [max(0,buoy_pnt_1[1]-self.wnd),min(480,buoy_pnt_1[1]+self.wnd)+1,max(0,buoy_pnt_1[0]-self.wnd),min(640,buoy_pnt_1[0]+self.wnd)+1]
        bds2 = [max(0,buoy_pnt_2[1]-self.wnd),min(480,buoy_pnt_2[1]+self.wnd)+1,max(0,buoy_pnt_2[0]-self.wnd),min(640,buoy_pnt_2[0]+self.wnd)+1]
        bds3 = [max(0,buoy_pnt_3[1]-self.wnd),min(480,buoy_pnt_3[1]+self.wnd)+1,max(0,buoy_pnt_3[0]-self.wnd),min(640,buoy_pnt_3[0]+self.wnd)+1]
        
        out_image = np.zeros((480, 640, 3), dtype=np.uint8)



        buoy1_img = img[bds1[0]:bds1[1],bds1[2]:bds1[3]]

        # temp_image1 = buoy1_img.copy()
        
        color1u = np.array([img[buoy_pnt_1[1],buoy_pnt_1[0],0]+self.thres[0],img[buoy_pnt_1[1],buoy_pnt_1[0],1]+self.thres[0],img[buoy_pnt_1[1],buoy_pnt_1[0],2]+self.thres[0]])
        color1b = np.array([img[buoy_pnt_1[1],buoy_pnt_1[0],0]-self.thres[0],img[buoy_pnt_1[1],buoy_pnt_1[0],1]-self.thres[0],img[buoy_pnt_1[1],buoy_pnt_1[0],2]-self.thres[0]])
        mask1 = cv2.inRange(buoy1_img, np.clip(color1b,0,255), np.clip(color1u,0,255))
        
        # temp_image1 = np.hstack([temp_image1, cv2.merge([mask1.copy(),mask1.copy(),mask1.copy()])])
        
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, self.kernel3)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, self.kernel5)
        out_image[bds1[0]:bds1[1],bds1[2]:bds1[3],0] = mask1

        # mask1_1 = cv2.inRange(buoy1_img, (0,200,200), (255,255,255))
        # temp_image1 = np.hstack([temp_image1, cv2.merge([mask1_1.copy(),mask1_1.copy(),mask1_1.copy()])])
        
        

        if (buoy_pnt_2 != [0,0]).all():
            buoy2_img = img[bds2[0]:bds2[1],bds2[2]:bds2[3]]
            # temp_image2 = buoy2_img.copy()
            color2u = np.array([img[buoy_pnt_2[1],buoy_pnt_2[0],0]+self.thres[1],img[buoy_pnt_2[1],buoy_pnt_2[0],1]+self.thres[1],img[buoy_pnt_2[1],buoy_pnt_2[0],2]+self.thres[1]])
            color2b = np.array([img[buoy_pnt_2[1],buoy_pnt_2[0],0]-self.thres[1],img[buoy_pnt_2[1],buoy_pnt_2[0],1]-self.thres[1],img[buoy_pnt_2[1],buoy_pnt_2[0],2]-self.thres[1]])
            mask2 = cv2.inRange(buoy2_img, np.clip(color2b,0,255), np.clip(color2u,0,255))
            # temp_image2 = np.hstack([temp_image2, cv2.merge([mask2.copy(),mask2.copy(),mask2.copy()])])
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, self.kernel3)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, self.kernel5)
            out_image[bds2[0]:bds2[1],bds2[2]:bds2[3],1] = mask2

            # mask2_1 = cv2.inRange(buoy2_img, (0,0,200), (255,255,255))
            # temp_image2 = np.hstack([temp_image2, cv2.merge([mask2_1.copy(),mask2_1.copy(),mask2_1.copy()])])

        
        if (buoy_pnt_3 != [0,0]).all():
            buoy3_img = img[bds3[0]:bds3[1],bds3[2]:bds3[3]]
            temp_image3 = buoy3_img.copy()
            color3u = np.array([img[buoy_pnt_3[1],buoy_pnt_3[0],0]+self.thres[2],img[buoy_pnt_3[1],buoy_pnt_3[0],1]+self.thres[2],img[buoy_pnt_3[1],buoy_pnt_3[0],2]+self.thres[2]])
            color3b = np.array([img[buoy_pnt_3[1],buoy_pnt_3[0],0]-self.thres[2],img[buoy_pnt_3[1],buoy_pnt_3[0],1]-self.thres[2],img[buoy_pnt_3[1],buoy_pnt_3[0],2]-self.thres[2]])
            mask3 = cv2.inRange(buoy3_img, np.clip(color3b,0,255), np.clip(color3u,0,255))
            
            mask3 = cv2.circle(mask3, (round(.45*mask3.shape[0]),round(.45*mask3.shape[1])), 9, 255, 6) 
            # temp_image3 = np.hstack([temp_image3, cv2.merge([mask3.copy(),mask3.copy(),mask3.copy()])])
            mask3 = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, self.kernel3)
            # mask3 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, self.kernel3)
            out_image[bds3[0]:bds3[1],bds3[2]:bds3[3],2] = mask3
            # out_image = cv2.circle(out_image, (buoy_pnt_3[0],buoy_pnt_3[1]), 9, (0,0,255), 6) 
            window_name = 'Image'
   


            # mask3_1 = cv2.inRange(buoy2_img, (0,200,0), (255,255,255))
            # temp_image3 = np.hstack([temp_image3, cv2.merge([mask3_1.copy(),mask3_1.copy(),mask3_1.copy()])])


        # total_image = np.vstack([temp_image1,temp_image2,temp_image3])
        # cv2.imwrite( "masking_image.png", total_image)
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

    def get_mask_old(self,img,img_num):
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
    # train data, number of ellipsoids, iterations
    def __init__(self,X,number_of_sources,iterations,threshold):
        self.iterations = iterations
        self.number_of_sources = number_of_sources
        self.X = X
        self.mu = None
        self.pi = None
        self.cov = None
        self.XY = None
        self.show_plot = False
        self.show_log_plot = True
        self.h = X.shape[0]
        self.w = X.shape[1]
        self.threshold = threshold
        

    def save_params(self, prefix):
        np.save(prefix + "median.npy", self.med)
        np.save(prefix + "mean.npy", self.mu)
        np.save(prefix + "covars.npy", self.cov)
        np.save(prefix + "weights.npy", self.pi)

    def load_params(self, prefix):
        self.med = np.load(prefix + "median.npy")
        self.mu = np.load(prefix + "mean.npy")
        self.cov = np.load(prefix + "covars.npy")
        self.pi = np.load(prefix + "weights.npy")
    

    """Define a function which runs for iterations, iterations"""
    def train(self):
        
        self.reg_cov = 1e-6*np.identity(len(self.X[0]))
        # coordinates of the data points
        # x,y = np.meshgrid(np.sort(self.X[:,0]),np.sort(self.X[:,1]))
        # self.XY = np.array([x.flatten(),y.flatten()]).T
        
        # self.mu=np.array([[ 73,  95,  98],
        # [113, 127, 104],
        # [172, 100, 136],
        # [ 90,  94,  97],
        # [150, 171, 140],
        # [108,  57, 136],
        # [129, 139,  67]])

        """ 1. Set the initial mu, covariance and pi values"""
        # self.mu = np.random.randint(min(self.X[:,0]),max(self.X[:,0]),size=(self.number_of_sources,len(self.X[0]))) 
        
        # This is a nxm matrix since we assume n sources (n Gaussians) where each has m dimensions
        # self.cov = np.zeros((self.number_of_sources,len(X[0]),len(X[0]))) 
        # # We need a nxmxm covariance matrix for each source since we have m features --> We create symmetric covariance matrices with ones on the digonal
        # for dim in range(len(self.cov)):
        #     np.fill_diagonal(self.cov[dim],5)

        self.cov = [ np.cov(self.X.T) for _ in range(self.number_of_sources) ]
        

        idxs = np.random.choice(self.h, self.number_of_sources, False)
        self.mu = np.array(self.X[idxs, :], dtype=np.float)

        

        self.pi = np.ones(self.number_of_sources)/self.number_of_sources # Are "Fractions"
        log_likelihoods = [] # In this list we store the log likehoods per iteration and plot them in the end to check if
        
        # print(self.mu)
        # print(self.cov)
        # print(np.shape(self.mu))
        # print(len(self.cov))
        # print(self.pi)
                                 # if we have converged
        # print(len(self.cov))
        # print(self.X)    
        """Plot the initial state"""    
        

        # for m,c in zip(self.mu,self.cov):
        #     c += self.reg_cov
        #     multi_normal = multivariate_normal(mean=m,cov=c)
        #     # ax0.contour(np.sort(self.X[:,0]),np.sort(self.X[:,1]), \
        #     #     multi_normal.pdf(self.XY).reshape(len(self.X),len(self.X)),colors='black',alpha=0.3)
        #     ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)
        # plt.show()

        
        flag = 1
        it = 0

        if self.show_plot:
            fig = plt.figure()
            ax0 = Axes3D(fig)

            scat=ax0.scatter(self.X[:,0],self.X[:,1],self.X[:,2])
            ax0.set_title('Initial state')

        for i in range(self.iterations):               
            print("Iteration {}".format(i))
            """E Step"""
            # number of samples * number of sources 
            r_ic = np.zeros((len(self.X),len(self.cov)))
            # if flag:
            #     print(np.shape(r_ic))
            #     flag = 0
            
            # print(np.shape(np.sum([pi_c*multivariate_normal(mean=mu_c,cov=cov_c).pdf(X) \
            #         for pi_c,mu_c,cov_c in zip(self.pi,self.mu,self.cov+self.reg_cov)],axis=0)))
          
            for m,co,p,r in zip(self.mu,self.cov,self.pi,range(len(r_ic[0]))):
                
                # co+=self.reg_cov
                # print(co)
                mn = multivariate_normal(mean=m,cov=co)
                # print(it)
                # if flag:
                #     # print(mn.pdf(self.X)) # This is a 500 length vector 
                #     flag = 0
                #     print(np.shape(r_ic))
                #     print(co)

                
                r_ic[:,r] = p*mn.pdf(self.X)/np.sum([pi_c*multivariate_normal(mean=mu_c,cov=cov_c).pdf(self.X) \
                    for pi_c,mu_c,cov_c in zip(self.pi,self.mu,self.cov+self.reg_cov)],axis=0)
                # print(r_ic)
                # temp[:,r] = r_ic[:,r]
            #     r_ic[:,r] = p*mn.pdf(self.X)
            # print(np.shape(r_ic))
            # print(np.shape(r_ic))
            # denom = np.sum(r_ic,axis=1)
            # denom_ = np.vstack((denom, denom, denom, denom, denom, denom, denom)) 
            # print(np.shape(r_ic))
            # # print(denom)
            # print(r_ic/denom_.T)
            
            """M Step"""

            # Calculate the new mean vector and new covariance matrices, based on the probable membership of the single x_i to classes c --> r_ic
            self.mu = []
            self.cov = []
            self.pi = []
            log_likelihood = []
            
            for c in range(len(r_ic[0])):

                m_c = np.sum(r_ic[:,c],axis=0)
                # print(m_c)
                # boo = self.X*r_ic[:,c].reshape(len(self.X),1)
                # # print(                     np.shape(boo))
                # val = np.hstack(( self.X[:,0].reshape(len(self.X),1)*r_ic[:,c].reshape(len(self.X),1),\
                #  self.X[:,1].reshape(len(self.X),1)*r_ic[:,c].reshape(len(self.X),1) ))
                # print(val-boo)
                # print(boo[:,0].reshape(len(self.X),1))
                # print(len(self.X[:,0]))
                mu_c = (1/m_c)*np.sum(self.X*r_ic[:,c].reshape(len(self.X),1),axis=0)
                # print(mu_c)
                self.mu.append(mu_c)

                # Calculate the covariance matrix per source based on the new mean
                self.cov.append(((1/m_c)*np.dot((np.array(r_ic[:,c]).reshape(len(self.X),1) \
                    *(self.X-mu_c)).T,(self.X-mu_c)))+self.reg_cov)
                
                self.pi.append(m_c/np.sum(r_ic))
            
            """Log likelihood"""
            log_likelihoods.append(np.log(np.sum([k*multivariate_normal(self.mu[i],self.cov[j]).pdf(self.X) \
                for k,i,j in zip(self.pi,range(len(self.mu)),range(len(self.cov)))])))

            for m,c in zip(self.mu,self.cov):
                c += self.reg_cov
                multi_normal = multivariate_normal(mean=m,cov=c)
                # ax0.contour(np.sort(self.X[:,0]),np.sort(self.X[:,1]), \
                #     multi_normal.pdf(self.XY).reshape(len(self.X),len(self.X)),colors='black',alpha=0.3)
                if self.show_plot:
                    ax0.scatter(m[0],m[1],m[2],c='red')
            # scat.set_offsets(self.mu)

            if self.show_plot:
                plt.pause(0.01)
            if i < self.iterations-1:
                for m,c in zip(self.mu,self.cov):
                    c += self.reg_cov
                    multi_normal = multivariate_normal(mean=m,cov=c)
                    # ax0.contour(np.sort(self.X[:,0]),np.sort(self.X[:,1]), \
                    #     multi_normal.pdf(self.XY).reshape(len(self.X),len(self.X)),colors='black',alpha=0.3)
                    if self.show_plot:

                        ax0.scatter(m[0],m[1],m[2],c='white')
                        e_vals, e_vecs = np.linalg.eig(c) # val, vecs
                        radii = np.sqrt(e_vals)
                        rotation = e_vecs 
                        center = m 
                        self.plot_ellipsoid_3d(center, radii, rotation, ax0)
            # else: 
            #     print("eabvhauet")
            #     for m,c in zip(self.mu,self.cov):
            #         c += self.reg_cov
            #         multi_normal = multivariate_normal(mean=m,cov=c)
            #         # ax0.contour(np.sort(self.X[:,0]),np.sort(self.X[:,1]), \
            #         #     multi_normal.pdf(self.XY).reshape(len(self.X),len(self.X)),colors='black',alpha=0.3)
            #         ax0.scatter(m[0],m[1],m[2],c='red',s=5)
           
        # fig = plt.figure(figsize=(10,10))
        # ax0 = fig.add_subplot(111)
        

        if self.show_log_plot:
            fig2 = plt.figure(figsize=(10,10))
            ax1 = fig2.add_subplot(111) 
            ax1.set_title('Log-Likelihood')
            ax1.plot(range(0,self.iterations,1),log_likelihoods)
            plt.show()



        # new_data = np.reshape(train_data, (sh[0], 1, 3))
    
        likelihoods = np.zeros((self.h, self.number_of_sources))
        # Compute likelihoods
        for i in range(self.number_of_sources):
            normal = multivariate_normal.pdf(self.X,
                                             self.mu[i],
                                             self.cov[i],
                                             allow_singular=True)
            pd = self.pi[i] * normal
            likelihoods[:, i] = pd

        med = likelihoods
        self.med = np.max(np.mean(med, axis=1))
        print(self.med)
    
    # """Predict the membership of an unseen, new datapoint"""
    # def predict(self,Y):
    #     # PLot the point onto the fittet gaussians
    #     fig3 = plt.figure()
    #     ax2 = fig3.add_subplot(111)
    #     ax2.scatter(self.X[:,0],self.X[:,1])
    #     for m,c in zip(self.mu,self.cov):
    #         multi_normal = multivariate_normal(mean=m,cov=c)
    #         ax2.contour(np.sort(self.X[:,0]),np.sort(self.X[:,1]),multi_normal.pdf(self.XY).\
    #             reshape(len(self.X),len(self.X)),colors='black',alpha=0.3)
    #         ax2.scatter(m[0],m[1],c='grey',zorder=10,s=100)
    #         ax2.set_title('Final state')
    #         for y in Y:
    #             ax2.scatter(y[0],y[1],c='orange',zorder=10,s=100)
    #     prediction = []        
    #     for m,c in zip(self.mu,self.cov):  
    #         #print(c)
    #         prediction.append(multivariate_normal(mean=m,cov=c).pdf(Y)/np.sum(\
    #             [multivariate_normal(mean=mean,cov=cov).pdf(Y) for mean,cov in zip(self.mu,self.cov)]))
    #     #plt.show()
    #     return prediction
         
 


    def apply_gmm(self,img):

        # if self._color:
            # img_s = cv2.cvtColor(img, self._color)
        
        y = 480
        x = 640

        data_img = np.reshape(img, (x*y, 3))

        prob = np.zeros((data_img.shape[0], self.number_of_sources))

        # Compute Probabilty
        for i in range(self.number_of_sources):
            normal = multivariate_normal.pdf(data_img,self.mu[i],self.cov[i],allow_singular=True)
            pd = self.pi[i] * normal
            prob[:, i] = pd

        # Reshape into image format
        loglikes = np.sum(prob, axis=1)
        prob_img = np.reshape(loglikes, (y, x))
        # print('Average threshold: ',np.mean(prob_img))
        # Find points above threshold
        out_image = np.zeros((y, x), dtype=np.uint8)
        out_image[prob_img > self.med*self.threshold] = 255
        # out_image[loglike_img > self.median*self.threshold] = 255        
        
        return out_image


class buoy_detector:
    def __init__(self):
        self.kernel3 = np.ones((3,3),np.uint8)
        self.kernel5 = np.ones((5,5),np.uint8)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    
    def add_points(self,img,pnts):
        for pnt in pnts:
            # print(pnt)
            if (pnt != [0,0]).all():
                img_point = (int(pnt[0]),int(pnt[1]))
                cv2.circle(img, img_point, 5, (50,50,50), 10)
                cv2.putText(img,str(img_point), img_point, self.font, .5,(50,50,50),1,cv2.LINE_AA)

        return img

    def detect(self,img):

        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, self.kernel3)
        contours, hierarchy = cv2.findContours(img.copy() ,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # contours = np.squeeze(contours)
        blank_mask = np.zeros((480,640,1),np.uint8)
        
        if contours is not None: 
            areas = np.zeros((len(contours),1))
            # heights = np.zeros((len(contours),1))
            pnts = np.zeros((len(contours),2))

            for contour_indx in range(len(contours)):
                con_temp = contours[contour_indx]

                area_temp = cv2.contourArea(con_temp)
                height = max(con_temp[:,0,1]) - min(con_temp[:,0,1])
                width = max(con_temp[:,0,0]) - min(con_temp[:,0,0])
                
                if area_temp > 20 & height < 80:# & height/width < 3.0:
                    areas[contour_indx] = area_temp
                    pnts[contour_indx,:] = np.mean(con_temp,axis=0)
            # print((areas>20).any())

            if (areas > 20).any():
                final_indx = np.argmax(areas)
                contour_final = contours[final_indx]
                pnt_final = pnts[final_indx]

                # print(contour_final.shape)        
                cv2.drawContours(blank_mask, [contour_final], 0, 255, 3)

                # cv2.imshow('result',blank_mask)
                # cv2.waitKey(-1)
                
                return pnt_final,blank_mask

            else:
                # print('failed1')
                return np.array([0,0]),blank_mask
        else:
            # print('failed2')
            return np.array([0,0]),blank_mask


        
        return img
