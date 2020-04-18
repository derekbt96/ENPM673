import numpy as np 
import cv2
from matplotlib import pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d import axes3d, Axes3D  
from functions import color_mask, color_data




def main():
	
	capture = cv2.VideoCapture('detectbuoy.avi')
	mask_gen = color_mask()

	ret, frame = capture.read()
	mask = mask_gen.get_mask(frame,0)
	color_seg1, color_seg2, color_seg3, color_segR = mask_gen.get_all_arrays(frame,mask)
	
	
	capture.release()
	
	
	fig = plt.figure()
	plt.subplot(411)
	plt.hist(color_seg1[:,0], bins=256, range=(0.0, 255.0), fc='b', ec='b')
	plt.hist(color_seg1[:,1], bins=256, range=(0.0, 255.0), fc='g', ec='g')
	plt.hist(color_seg1[:,2], bins=256, range=(0.0, 255.0), fc='r', ec='r')
	plt.title('Image Histogram Buoy 1')

	plt.subplot(412)
	plt.hist(color_seg2[:,0], bins=256, range=(0.0, 255.0), fc='b', ec='b')
	plt.hist(color_seg2[:,1], bins=256, range=(0.0, 255.0), fc='g', ec='g')
	plt.hist(color_seg2[:,2], bins=256, range=(0.0, 255.0), fc='r', ec='r')
	plt.title('Image Histogram Buoy 2')

	plt.subplot(413)
	plt.hist(color_seg3[:,0], bins=256, range=(0.0, 255.0), fc='b', ec='b')
	plt.hist(color_seg3[:,1], bins=256, range=(0.0, 255.0), fc='g', ec='g')
	plt.hist(color_seg3[:,2], bins=256, range=(0.0, 255.0), fc='r', ec='r')
	# plt.hist(.7*color_seg3[:,1]+.3*color_seg3[:,2], bins=256, range=(0.0, 255.0), fc='k', ec='k')
	plt.title('Image Histogram Buoy 3')
	
	plt.subplot(414)
	plt.hist(color_segR[:,0], bins=256, range=(0.0, 255.0), fc='b', ec='b')
	plt.hist(color_segR[:,1], bins=256, range=(0.0, 255.0), fc='g', ec='g')
	plt.hist(color_segR[:,2], bins=256, range=(0.0, 255.0), fc='r', ec='r')
	# plt.hist(color_segR[:,1], bins=256, range=(0.0, 255.0), fc='k', ec='k')
	plt.title('Image Histogram Remainder')

	plt.show()
	


main()



# cd Documents/Documents/Aerospace/ENPM673/ENPM673/Project_3