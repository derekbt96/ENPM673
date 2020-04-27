import cv2
import numpy as np 
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import os
dirpath = os.getcwd()
fx ,fy ,cx ,cy ,G_camera_image, LUT = ReadCameraModel('./model')
# iterate over all images
it = 1
img1 = 0
img_orig1 = 0

for subdir, dirs, files in os.walk(dirpath + '/stereo/centre'):
	files.sort()
	for file in files:
		filepath = subdir + os.sep + file

		if filepath.endswith(".jpg") or filepath.endswith(".pgm") or filepath.endswith(".png") or filepath.endswith(".ppm"):

			imgname = filepath
			# load image
			img = cv2.imread(imgname,0)
			img = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
			img_orig = UndistortImage(img, LUT)
			img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
			
			if (it == 1): 
				img_orig1 = img_orig
				img1 = img 
				it = 2
				continue 
			img2 = img
			img_orig2 = img_orig 

			# orb = cv2.ORB_create(nfeatures=2000)
			# kp, des = orb.detectAndCompute(gray_img, None)

			# kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

			orb = cv2.ORB_create(nfeatures=500)
			kp1, des1 = orb.detectAndCompute(img1, None)
			kp2, des2 = orb.detectAndCompute(img2, None)

			# cv2.NORM_HAMMING for ORB
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
			matches = bf.match(des1, des2)
			matches = sorted(matches, key=lambda x: x.distance)# draw first 50 matches
			match_img = cv2.drawMatches(img_orig1, 
				kp1, img_orig2, kp2, matches[:50], None)

			img1 = img2 
			img_orig1 = img_orig2

			# cv2.imshow('ORB', kp_img)
			cv2.imshow('Matches', match_img)
			cv2.waitKey(1000)
