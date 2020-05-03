import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import cv2

def Fundamental(points_f1, points_f2):

    A1 = np.multiply(points_f1, np.tile(points_f2[:,0],(3,1)).T)
    A2 = np.multiply(points_f1, np.tile(points_f2[:,1],(3,1)).T)
    A3 = points_f1
    A = np.hstack((np.hstack((A1, A2)), A3))

    U, S, V = np.linalg.svd(A)
    F = np.reshape(V[:,-1], (3,3))
    
    U, S, V = np.linalg.svd(F)
    S = np.diag(S)
    # enforce rank 2 condition
    S[2,2] = 0
    # recalculate Fundamental matrix
    F = np.matmul(np.matmul(U, S), np.transpose(V)) 

    return F

def NormalizedFundamental(points_f1, points_f2):

    l = len(points_f2)

    centroid_f1 = np.mean(points_f1, 0)
    centroid_f2 = np.mean(points_f2, 0)
    # Recentre feature points
    f1_centred = points_f1 - np.tile(centroid_f1, (l,1))
    f2_centred = points_f2 - np.tile(centroid_f2, (l,1))
    """
    standard deviation. The final multiplication is because 
    python while calculating variance divided it by n and not n-1
    """
    s_f1 = np.sqrt(np.var(f1_centred, axis=0)*(l/(l-1)))
    s_f2 = np.sqrt(np.var(f2_centred, axis=0)*(l/(l-1)))
    # Transformation matrix
    Ta = np.matmul([
        [1/s_f1[0], 0, 0], 
        [0, 1/s_f1[1], 0], 
        [0, 0, 1]],

        [[1, 0, -centroid_f1[0]], 
        [0, 1, -centroid_f1[1]], 
        [0, 0, 1]])
    
    Tb = np.matmul([
        [1/s_f2[0], 0, 0], 
        [0, 1/s_f2[1], 0], 
        [0, 0, 1]],

        [[1, 0, -centroid_f2[0]], 
        [0, 1, -centroid_f2[1]], 
        [0, 0, 1]])
    # Normalized points
    Normalized_f1 = np.matmul(Ta, points_f1.T).T
    Normalized_f2 = np.matmul(Tb, points_f2.T).T

    F_norm = Fundamental(Normalized_f1, Normalized_f2)
    nF = np.matmul(np.matmul(np.transpose(Tb), F_norm), Ta)

    return nF

def RansacFundamental(points_f1, points_f2): 

    l = len(points_f2)
    
    points_f1 = np.reshape(points_f1[:,0], (l,2))
    points_f2 = np.reshape(points_f2[:,0], (l,2))

    # Change to homogeneous coordinates
    points_f1 = np.hstack((points_f1, np.ones((l,1))))
    points_f2 = np.hstack((points_f2, np.ones((l,1))))
    
    # Initialize Fundamental matrix 
    best_F = np.zeros((3,3))

    # threshold for model convergence 
    thresh = .1
    # Number of points selected for a given iteration (8-point algo)
    it_points = 8
    # total number of iterations for which ransac should run
    total_it = 1000
    max_inliers = 0
    min_error = 1000000

    for it in range(total_it):
        # print(it)
        rand_index = np.random.choice(l, it_points, replace=True)

        nF = NormalizedFundamental(points_f1[rand_index], points_f2[rand_index])
        
        epipolar_constraint = np.sum(np.multiply(points_f2, np.transpose(np.matmul(nF, np.transpose(points_f1)))), 1)
        current_inliers = len(np.where(abs(epipolar_constraint) < thresh)[0])

        if (current_inliers > max_inliers):
            best_F = nF
            max_inliers = current_inliers
            # print(max_inliers)


        # epipolar = np.sum(np.multiply(points_f2, np.transpose(np.matmul(nF, np.transpose(points_f1)))), 1)
        # mean_error = np.mean(abs(epipolar))
        # if (mean_error < min_error):
        #     best_F = nF
        #     min_error = mean_error
        #     inliers = epipolar[abs(epipolar) < thresh]
        #     print(len(inliers),points_f1.shape[0])


    # print(max_inliers,points_f1.shape[0])

    error = np.sum(np.multiply(points_f2, np.transpose(np.matmul(best_F, np.transpose(points_f1)))), 1)
    indices = np.argsort(abs(error))

    # Pick out the least erroneous k inliers
    k = 30

    inliers_f1 = points_f1[indices[:k]]
    inliers_f2 = points_f2[indices[:k]]
    # inliers_f1 = points_f1
    # inliers_f2 = points_f2


    return best_F, inliers_f1, inliers_f2

def EpipolarLines(img_f1, points_f1, img_f2, points_f2, F):
    h, w, d = np.shape(img_f1)

    top_left = np.array([1,1,1])
    bot_left = np.array([1,h,1])
    top_rt = np.array([w,1,1])
    bot_rt = np.array([w,h,1])

    # Vertical line on the left side of any of the two images 
    line_left = np.cross(top_left, bot_left)
    # Vertical line on the right side of any of the two images
    line_right = np.cross(top_rt, bot_rt)
    
    img_f1 = cv2.UMat(img_f1)
    img_f2 = cv2.UMat(img_f2)

    for it in range(len(points_f1)):
        # epipolar line in the left image 
        l = np.matmul(F, np.reshape(points_f1[it,:],(3,1)))
        # epipolar line in the right image
        l_ = np.matmul(F.T, np.reshape(points_f2[it,:],(3,1)))

        f1_p1 = np.cross(l.flatten(), line_left)
        f1_p2 = np.cross(l.flatten(), line_right)
        f1_p1 /= f1_p1[2]
        f1_p2 /= f1_p2[2]

        f2_p1 = np.cross(l_.flatten(), line_left)
        f2_p2 = np.cross(l_.flatten(), line_right)
        f2_p1 /= f2_p1[2]
        f2_p2 /= f2_p2[2]

        img_f1 = cv2.line(
            img_f1, 
            (int(f1_p1[0]), int(f1_p1[1])), 
            (int(f1_p2[0]), int(f1_p2[1])), 
            (0,0,255), 
            1) 
        img_f1 = cv2.circle(img_f1, 
            (int(points_f1[it,0]), int(points_f1[it,1])), 
            3, (255,0,0), 2) 

        img_f2 = cv2.line(
            img_f2, 
            (int(f2_p1[0]), int(f2_p1[1])), 
            (int(f2_p2[0]), int(f2_p2[1])), 
            (0,0,255), 
            1) 
        img_f2 = cv2.circle(img_f2, 
            (int(points_f2[it,0]), int(points_f2[it,1])), 
            3, (255,0,0), 2) 
    
    return img_f1, img_f2
        

def plotCoordinates(pnts):
    plt.figure(figsize=(6,6))
    print(pnts.shape)
    num_set = int(pnts.shape[1]/2)
    print(num_set)
    for i in range(num_set):
        plt.scatter(pnts[:,2*i], pnts[:,2*i+1])#,s=.6)
    plt.grid(True)
    plt.show()



class camera_movement():
    def __init__(self):
        
        self.pos = np.zeros((3,1))
        self.R = np.identity(3)
        self.X_log = np.zeros((1,3))
        

    def update_pos(self,r,t):
        self.R = np.matmul(r,self.R)
        self.pos = np.hstack(self.pos) + np.matmul(self.R,t)
        self.X_log = np.vstack([self.X_log,self.pos])
        

    def plot(self):
        plt.figure(figsize=(6,6))
        plt.plot([self.X_log][:,0], self.X_log[:,2])
        plt.grid(True)
        plt.show()



def getCameraPose(F,K,p_old,p_new):

    # Get essential matrix
    E = np.matmul(np.matmul(K.T,F),K)
    
    U, S, V = np.linalg.svd(E)
    temp = np.array([[1,0,0],[0,1,0],[0,0,0]])
    E = np.matmul(np.matmul(U,temp),V.T)
    
    # Get Camera Pose
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    
    # print(U)
    C1 = U[:,2]
    C2 = -U[:,2]
    C3 = U[:,2]
    C4 = -U[:,2]
    
    R1 = np.matmul(np.matmul(U,W),V.T)
    R2 = np.matmul(np.matmul(U,W),V.T)
    R3 = np.matmul(np.matmul(U,W.T),V.T)
    R4 = np.matmul(np.matmul(U,W.T),V.T)
    
    # print(R1)
    # print(R2)
    # print(R3)
    # print(R4)
    
    return U[:,2],R3

    # P1 = np.matmul(np.matmul(K,R1),np.hstack([np.identity(3), np.vstack(-C1)]))

    p_old = np.hstack([np.squeeze(p_old),np.ones((p_old.shape[0],1))])
    p_new = np.hstack([np.squeeze(p_new),np.ones((p_new.shape[0],1))])
    
    Kinv = np.linalg.inv(K)

    x_old = np.matmul(Kinv,p_old.T)
    x_new = np.matmul(Kinv,p_new.T)
    
    # pnts1 = (x_new.T - C1)
    # pnts2 = (x_new.T - C2)
    # pnts3 = (x_new.T - C3)
    # pnts4 = (x_new.T - C4)

    # depth1 = np.matmul(R1[2,:],pnts1.T)
    # depth2 = np.matmul(R2[2,:],pnts2.T)
    # depth3 = np.matmul(R3[2,:],pnts3.T)
    # depth4 = np.matmul(R4[2,:],pnts4.T)
    
    # print([np.sum(np.mean(depth1)),np.sum(np.mean(depth2)),np.sum(np.mean(depth3)),np.sum(np.mean(depth4))])
    # print(C1[0],np.mean(depth1))
    # print(C2[0],np.mean(depth2))
    # print(C3[0],np.mean(depth3))
    # print(C4[0],np.mean(depth4))

