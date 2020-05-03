import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import os
from functions import RansacFundamental, EpipolarLines, getCameraPose, Linear,checkCheirality, camera_pose
dirpath = os.getcwd()

fx ,fy ,cx ,cy ,G_camera_image, LUT = ReadCameraModel('./model')
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

# iterate over all images
it = -80
img1 = 0
img_orig1 = 0

orb_kp1 = None
orb_des1 = None

# Initialize Orb detector and BF matcher
orb = cv2.ORB_create(nfeatures=400,patchSize=51)
# print(orb.patchSize)
# orb = cv2.SIFT_create(nfeatures=400)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
pose = camera_pose()

for subdir, dirs, files in os.walk(dirpath + '/stereo/centre'):
    files.sort()
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".jpg") or filepath.endswith(".pgm") or filepath.endswith(".png") or filepath.endswith(".ppm"):

            if it < 1:
                it += 1
                continue

            # load image
            img = cv2.imread(filepath,0)
            img = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
            img_orig = UndistortImage(img, LUT)
            img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
            
            if (it == 1): 
                img_orig1 = img_orig
                img1 = img 
                it = 2
                orb_kp1, orb_des1 = orb.detectAndCompute(img1, None)
                continue 

            it += 1
            img2 = img
            img_orig2 = img_orig 

            # find orb features 
            # kp1, des1 = orb.detectAndCompute(img1, None)
            kp1 = orb_kp1
            des1 = orb_des1

            kp2, des2 = orb.detectAndCompute(img2, None)

            # Feature matching: cv2.NORM_HAMMING for ORB
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            # matches = matches[:50] # draw first 50 matches
            match_img = cv2.drawMatches(img_orig1, kp1, img_orig2, kp2, matches, None)
            
            img1 = img2 
            img_orig1 = img_orig2
            orb_kp1 = kp2
            orb_des1 = des2

            points_f1 = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            points_f2 = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

            l = len(points_f2)
    
            points_f1 = np.reshape(points_f1[:,0], (l,2))
            points_f2 = np.reshape(points_f2[:,0], (l,2))

            # Change to homogeneous coordinates
            points_f1 = np.hstack((points_f1, np.ones((l,1))))
            points_f2 = np.hstack((points_f2, np.ones((l,1))))

            F, inliers_f1, inliers_f2 = RansacFundamental(points_f1, points_f2)
            
            # draw epipolar lines
            # img_f1, img_f2 = EpipolarLines(img_orig1, inliers_f1, img_orig2, inliers_f2, F)


            # These are the possible transforms between frame1 (f1) and frame2 (f2)
            T = getCameraPose(F, K, points_f1, points_f2)
            # X: 4 sets of 3D points (from triangulation) from 4 camera poses
            X = np.zeros((4,l,3))

            T_1 = np.identity(4)[:3,:] # 3x4

            for i in range(4):
                T_2 = T[0,:,:]
                X[i,:,:] = Linear(K, T_1, T_2, points_f1, points_f2)
            # R = T[:,:,:3]
            # t = T[:,:,3]
            # print(np.shape(R))
            # print(np.shape(t))
            # print(R)
            # print(t)

            # Final transform and points in front of camera
            R_final, t_final, X_final = checkCheirality(T, X)
            # print(R_final)
            # print(t_final)
            # print(np.shape(X_final))


            # Log camera movement
            # Calculate pose of the camera relative to the first camera pose
            pose.update(R_final, t_final, X_final)
            

            # break
            # cv2.imshow('img', img)
            # cv2.imshow('Epipolar lines', img_f1)
            match_img = cv2.resize(match_img, (1000, 400), interpolation = cv2.INTER_AREA)
            cv2.imshow('Epipolar lines', match_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

pose.plot()