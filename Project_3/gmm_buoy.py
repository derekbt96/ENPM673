import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import multivariate_normal
from functions import color_mask
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import cv2

class GMM:
    # train data, number of ellipsoids, iterations
    def __init__(self, X, distributions, iterations):
        self.it = iterations
        self.k = distributions
        self.X = X
        self.h = X.shape[0]
        self.w = X.shape[1]
        self.mu = None
        self.cov = None
        self.pi = None
        self.plt = True

    def plot_ellipsoid_3d(self, mu, cov, ax):

        e_vals, e_vecs = np.linalg.eig(cov) # val, vecs
        radius = np.sqrt(e_vals)
        rotation = e_vecs 
        center = mu

        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        theta = np.tile(v, (100, 1)).T
        phi = np.tile(u, (100, 1))
        # unrotated ellipsoid 
        # source: https://www.mathworks.com/matlabcentral/answers/86921-plot-an-ellipsoid-with-matlab
        # https://kittipatkampa.wordpress.com/2011/08/04/plot-3d-ellipsoid/
        x0 = radius[0]*np.multiply(np.sin(theta), np.cos(phi))
        y0 = radius[1]*np.multiply(np.sin(theta), np.sin(phi))
        z0 = radius[2]*np.cos(theta)

        a = np.kron( rotation[:,0].reshape(3,1), x0 )
        b = np.kron( rotation[:,1].reshape(3,1), y0 ) 
        c = np.kron( rotation[:,2].reshape(3,1), z0 )

        data = a + b + c;
        n = data.shape[1]
        
        x = data[0:n,:] + center[0]
        y = data[n:2*n,:] + center[1]
        z = data[2*n:3*n,:] + center[2]

        # print(np.shape(z))
        
        ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color='#2980b9', alpha=0.2)

    def train(self):

        # Randomly initialize the mean, covaraiance matrix and weights(pi) for each distribution
        idxs = np.random.choice(self.h, self.k, False)
        self.mu = np.array(self.X[idxs, :], dtype=np.float)
        
        self.safe_cov = 1e-6*np.identity(3)
        self.cov = [np.cov(self.X.T) for _ in range(self.k)]

        self.pi = np.ones(self.k)/self.k 

        # To store the log-likelihood to check convergence 
        log_likelihoods = [] 

        fig = plt.figure()
        ax0 = Axes3D(fig)
        # scatter input data 
        ax0.scatter(self.X[:,0], self.X[:,1], self.X[:,2], color=np.flip(self.X, axis=1)/255.0)
        ax0.set_xlabel('B')
        ax0.set_ylabel('G')
        ax0.set_zlabel('R')
        ax0.set_title('Input points and GMM')

        for it in range(self.it):

            print("Iteration {}".format(it))

            """E Step"""
            r_ic = np.zeros((self.h, self.k)) # number of input samples x k 
            
            for m,c,p,r in zip( self.mu, self.cov, self.pi, range(self.k) ):
               
                mn = multivariate_normal(mean=m,cov=c+self.safe_cov)
                r_ic[:,r] = p*mn.pdf(self.X)

            denom = np.sum(r_ic, axis=1)
            denom = np.tile(denom, (self.k, 1))
            r_ic = r_ic/denom.T
            
            """M Step"""
            self.mu = []
            self.cov = []
            self.pi = []
            
            # Update the mean, covariance, and weights(pi's)
            for c in range(self.k):

                m_c = np.sum(r_ic[:,c], axis=0)
                
                mu_c = (1/m_c) * np.sum( self.X * r_ic[:,c].reshape(self.h,1), axis=0 )
                self.mu.append(mu_c)

                cov_c = (1/m_c) * np.dot( (r_ic[:,c].reshape(self.h,1) * (self.X-mu_c)).T, \
                    (self.X-mu_c) )
                self.cov.append(cov_c+self.safe_cov)

                pi_c = m_c/np.sum(r_ic)
                self.pi.append(pi_c)
            
            log_likelihoods.append( \
                np.log( np.sum( [p*multivariate_normal(self.mu[m], self.cov[c]).pdf(self.X) \
                for p,m,c in zip(self.pi, range(self.k), range(self.k)) ] ) ) \
                )

            # Plot the means or the center point of gaussians 
            mu_plot = np.array(self.mu)
            ax0.scatter(mu_plot[:,0], mu_plot[:,1], mu_plot[:,2], c='black')

            plt.pause(0.01)

            if it < self.it-1:
                ax0.scatter(mu_plot[:,0], mu_plot[:,1], mu_plot[:,2], c='white')
                
        for m,c in zip(self.mu,self.cov):
     
            self.plot_ellipsoid_3d(m, c, ax0)

        fig2 = plt.figure()
        ax1 = fig2.add_subplot(111) 
        ax1.set_title('Log-Likelihood')
        ax1.plot(range(0,self.it,1),log_likelihoods)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Log value')
        if self.plt:
            plt.show()
    
    def predict(self,Y):

        prediction = []        
        for m,c in zip(self.mu,self.cov):  
            prediction.append( \
                multivariate_normal(mean=m, cov=c).pdf(Y)/np.sum( \
                [multivariate_normal(mean=mean, cov=cov).pdf(Y) \
                for mean, cov in zip(self.mu, self.cov)])\
                )
        return prediction

def main():
    # get input data 
    capture = cv2.VideoCapture('detectbuoy.avi')
    mask_gen = color_mask()

    ret, frame = capture.read()
    mask = mask_gen.get_mask()

    # right middle left
    HSV = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
    color_seg1, color_seg2, color_seg3, color_segR = mask_gen.get_all_arrays(HSV,mask)
    # input data
#     X = np.vstack((color_seg1, color_seg2, color_seg3))
    X = color_seg3

    distributions = 3
    iterations = 40

    gmm = GMM(X, distributions, iterations)    
    gmm.train()

main()
