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
	mask = mask_gen.get_mask()
	color_seg1, color_seg2, color_seg3, color_segR = mask_gen.get_all_arrays(frame,mask)
	
	fig = plt.figure()
	ax = Axes3D(fig)
	

	ax.scatter(color_seg1[:,0], color_seg1[:,1], color_seg1[:,2], c = (.8,.8,0))
	ax.scatter(color_seg2[:,0], color_seg2[:,1], color_seg2[:,2], c = (1,0,0))
	ax.scatter(color_seg3[:,0], color_seg3[:,1], color_seg3[:,2], c = (0,1,0))
	ax.scatter(color_segR[::50,0], color_segR[::50,1], color_segR[::50,2], c = (.6,.6,.6))
	
	ax.set_xlabel('B')
	ax.set_ylabel('G')
	ax.set_zlabel('R')
	ax.view_init(elev=70., azim=-150)
	plt.show()

	capture.release()
	mask_gen.cap.release()

	

main()

