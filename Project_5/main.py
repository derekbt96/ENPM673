import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import os
from functions import RansacFundamental, EpipolarLines, getCameraPose, camera_movement
dirpath = os.getcwd()

fx ,fy ,cx ,cy ,G_camera_image, LUT = ReadCameraModel('./model')
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

# iterate over all images
it = -20
img1 = 0
img_orig1 = 0

orb_kp1 = None
orb_des1 = None

# Initialize Orb detector and BF matcher
orb = cv2.ORB_create(nfeatures=400,patchSize=51)
# print(orb.patchSize)
# orb = cv2.SIFT_create(nfeatures=400)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
pose = camera_movement()

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
            

            # cv2.imshow('Matches', match_img)
            # cv2.waitKey(100)

            points_f1 = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            points_f2 = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

            F, inliers_f1, inliers_f2 = RansacFundamental(points_f1, points_f2)
            
            # draw epipolar lines
            img_f1, img_f2 = EpipolarLines(img_orig1, inliers_f1, img_orig2, inliers_f2, F)
            
            # Compute essential matrix
            t,R = getCameraPose(F, K, points_f1, points_f2)

            # Log camera movement
            pose.update_pos(R,t)
            

            # break
            cv2.imshow('Epipolar lines', img_f1)
            # cv2.imshow('Epipolar lines', match_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# pose.plot()