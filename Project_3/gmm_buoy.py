# ## Initialisation

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # %matplotlib inline
# ## Assignment Stage

# def assignment(df, centroids):
#     for i in centroids.keys():
#         # sqrt((x1 - x2)^2 - (y1 - y2)^2)
#         df['distance_from_{}'.format(i)] = (
#             np.sqrt(
#                 (df['x'] - centroids[i][0]) ** 2
#                 + (df['y'] - centroids[i][1]) ** 2
#             )
#         )
#     centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
#     df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
#     df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
#     df['color'] = df['closest'].map(lambda x: colmap[x])
#     return df

# def update(k):
#     for i in centroids.keys():
#     	# mean of all x's and y's that are categorized in the same cluster 
#         centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
#         centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
#     return k

# df = pd.DataFrame({
# 	'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
# 	'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
# })


# np.random.seed(200)
# k = 3
# # centroids[i] = [x, y]
# centroids = {
# 	i+1: [np.random.randint(0, 80), np.random.randint(0, 80)]
# 	for i in range(k)
# }
# print(centroids)

# fig = plt.figure(figsize=(5, 5))
# plt.scatter(df['x'], df['y'], color='k')
# colmap = {1: 'r', 2: 'g', 3: 'b'}
# for i in centroids.keys():
# 	plt.scatter(*centroids[i], color=colmap[i])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# plt.show()



# df = assignment(df, centroids)
# print(df.head())

# fig = plt.figure(figsize=(5, 5))
# plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colmap[i])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# plt.show()

# ## Update Stage

# import copy

# old_centroids = copy.deepcopy(centroids)

# centroids = update(centroids)
# print('update')
# print(centroids)
# fig = plt.figure(figsize=(5, 5))
# ax = plt.axes()
# plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colmap[i])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# for i in old_centroids.keys():
#     old_x = old_centroids[i][0]
#     old_y = old_centroids[i][1]
#     dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
#     dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
#     ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc=colmap[i], ec=colmap[i])
# plt.show()
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from scipy.stats import multivariate_normal
from functions import color_mask
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import cv2
# 0. Create dataset # XY, labels
# X,Y = make_blobs(cluster_std=1.5,random_state=20,n_samples=500,centers=3)

# # Stratch dataset to get ellipsoid data
# X = np.dot(X,np.random.RandomState(0).randn(2,2))

# roi = roi_data(1)
# buoy_rt, buoy_mt, buoy_lt = roi.train_data()


capture = cv2.VideoCapture('detectbuoy.avi')
mask_gen = color_mask()

ret, frame = capture.read()
mask = mask_gen.get_mask()
# right middle left
color_seg1, color_seg2, color_seg3, color_segR = mask_gen.get_all_arrays(frame,mask)

# input data
X = np.vstack((color_seg1, color_seg2, color_seg3))
h, w = np.shape(X)
class GMM:
    # train data, number of ellipsoids, iterations
    def __init__(self,X,number_of_sources,iterations):
        self.iterations = iterations
        self.number_of_sources = number_of_sources
        self.X = X
        self.mu = None
        self.pi = None
        self.cov = None
        self.XY = None
        
    

    """Define a function which runs for iterations, iterations"""
    def run(self):
        # A small threshold that adds in a small covariance matrix with the actual covariance mat
        # This is done so that the final covariance matrix is invertible even if the actual covariance matrix is singular
        self.reg_cov = 1e-6*np.identity(len(self.X[0]))
        # coordinates of the data points
        # x,y = np.meshgrid(np.sort(self.X[:,0]),np.sort(self.X[:,1]))
        # self.XY = np.array([x.flatten(),y.flatten()]).T
        
 #        self.mu=np.array([[ 73,  95,  98],
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

        self.cov = [ np.cov(X.T) for _ in range(self.number_of_sources) ]
        

        idxs = np.random.choice(h, self.number_of_sources, False)
        self.mu = np.array(X[idxs, :], dtype=np.float)

        

        self.pi = np.ones(self.number_of_sources)/self.number_of_sources # Are "Fractions"
        log_likelihoods = [] # In this list we store the log likehoods per iteration and plot them in the end to check if
        
        print(self.mu)
        print(self.cov)
        print(np.shape(self.mu))
        print(len(self.cov))
        print(self.pi)
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

                
                r_ic[:,r] = p*mn.pdf(self.X)/np.sum([pi_c*multivariate_normal(mean=mu_c,cov=cov_c).pdf(X) \
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
            log_likelihoods.append(np.log(np.sum([k*multivariate_normal(self.mu[i],self.cov[j]).pdf(X) \
                for k,i,j in zip(self.pi,range(len(self.mu)),range(len(self.cov)))])))

            for m,c in zip(self.mu,self.cov):
                c += self.reg_cov
                multi_normal = multivariate_normal(mean=m,cov=c)
                # ax0.contour(np.sort(self.X[:,0]),np.sort(self.X[:,1]), \
                #     multi_normal.pdf(self.XY).reshape(len(self.X),len(self.X)),colors='black',alpha=0.3)
                ax0.scatter(m[0],m[1],m[2],c='red')
            # scat.set_offsets(self.mu)

            plt.pause(0.01)
            if i < self.iterations-1:
                for m,c in zip(self.mu,self.cov):
                    c += self.reg_cov
                    multi_normal = multivariate_normal(mean=m,cov=c)
                    # ax0.contour(np.sort(self.X[:,0]),np.sort(self.X[:,1]), \
                    #     multi_normal.pdf(self.XY).reshape(len(self.X),len(self.X)),colors='black',alpha=0.3)
                    ax0.scatter(m[0],m[1],m[2],c='white')
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
        


        fig2 = plt.figure(figsize=(10,10))
        ax1 = fig2.add_subplot(111) 
        ax1.set_title('Log-Likelihood')
        ax1.plot(range(0,self.iterations,1),log_likelihoods)
        plt.show()
    
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
         
    
    
# # GMM = GMM(X,3,50)     
# # GMM.run()
# # GMM.predict([[0.5,0.5]])



# gmm_r = GMM(input sampels, train = True, train_data = buoy_rt)
# print(np.shape(input_samples))
gmm = GMM(X, 7, 40)    
gmm.run()
