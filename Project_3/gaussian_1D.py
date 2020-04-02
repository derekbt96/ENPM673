import numpy as np 
import cv2
from matplotlib import pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d import axes3d, Axes3D  
from functions import color_mask, color_data
from scipy.stats import multivariate_normal



def main():
	
	capture = cv2.VideoCapture('detectbuoy.avi')
	mask_gen = color_mask()

	ret, frame = capture.read()
	capture.release()
	# mask = mask_gen.get_mask()
	# color_seg1, color_seg2, color_seg3, color_segR = mask_gen.get_all_arrays(frame,mask)
	

	frame_arr = np.reshape(frame, (640*480, 3))

	data_1 = frame_arr[:,2]
	data_2 = frame_arr[:,1]
	data_3 = .7 * frame_arr[:,1] + .3*frame_arr[:,2]

	

	likelihoods = np.zeros((640*480,3))


	mean = [240,253,237]
	covariance = [30,10,10]

	print(frame_arr.shape)
	# Compute likelihoods
	
		
	
	Bp = multivariate_normal.pdf(data_1,mean[0],covariance[0],allow_singular=True)
	Gp = multivariate_normal.pdf(data_2,mean[1],covariance[1],allow_singular=True)
	Rp = multivariate_normal.pdf(data_3,mean[2],covariance[2],allow_singular=True)
	
	
	# fig = plt.figure()
	# plt.hist(Rp, bins=100, range=(0.0, max(Rp)), fc='r', ec='r')
	# plt.show()
	
	# print(max(Bp))
	# print(max(Gp))
	# print(max(Rp))

	thresholds = [.06, .1, .1]

	Bp = np.reshape(Bp, (480,640))
	Gp = np.reshape(Gp, (480,640))
	Rp = np.reshape(Rp, (480,640))
	

	# # Find points above threshold
	out_image1 = np.zeros((480, 640), dtype=np.uint8)
	out_image1[Bp > thresholds[0]] = 255
	cv2.imshow('result1',out_image1)
	
	out_image2 = np.zeros((480, 640), dtype=np.uint8)
	out_image2[Gp > thresholds[1]] = 255
	cv2.imshow('result2',out_image2)
	
	out_image3 = np.zeros((480, 640), dtype=np.uint8)
	out_image3[Rp > thresholds[2]] = 255
	cv2.imshow('result3',out_image3)
	
	
	
	cv2.waitKey(-1)



	
	
	

main()

