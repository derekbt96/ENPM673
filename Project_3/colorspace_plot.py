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
	HSV = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
	color_seg1, color_seg2, color_seg3, color_segR = mask_gen.get_all_arrays(frame,mask)
	
	fig = plt.figure()
	ax = Axes3D(fig)
		
	freq_color = 5
	freq_R = 150


	ax.scatter(color_seg1[::freq_color,0], color_seg1[::freq_color,1], color_seg1[::freq_color,2], c = (.8,.8,0))
	ax.scatter(color_seg2[::freq_color,0], color_seg2[::freq_color,1], color_seg2[::freq_color,2], c = (1,0,0))
	ax.scatter(color_seg3[::freq_color,0], color_seg3[::freq_color,1], color_seg3[::freq_color,2], c = (0,1,0))
	ax.scatter(color_segR[::freq_R,0], color_segR[::freq_R,1], color_segR[::freq_R,2], c = (.6,.6,.6))
	
	

	# temp = np.vstack([color_seg1, color_seg2, color_seg3])
	# ax.scatter(color_seg1[::freq_color,0], color_seg1[::freq_color,1], color_seg1[::freq_color,2], c = np.flip(color_seg1[::freq_color,:],axis=1)/255.0)
	# ax.scatter(color_seg2[::freq_color,0], color_seg2[::freq_color,1], color_seg2[::freq_color,2], c = np.flip(color_seg2[::freq_color,:],axis=1)/255.0)
	# ax.scatter(color_seg3[::freq_color,0], color_seg3[::freq_color,1], color_seg3[::freq_color,2], c = np.flip(color_seg3[::freq_color,:],axis=1)/255.0)
	# ax.scatter(color_segR[::freq_R,0], color_segR[::freq_R,1], color_segR[::freq_R,2], c = np.flip(color_segR[::freq_R,:],axis=1)/255.0)
	

	# ax.set_xlabel('H')
	# ax.set_ylabel('G')
	# ax.set_zlabel('R')
	ax.view_init(elev=70., azim=-150)
	plt.show()

	capture.release()
	mask_gen.cap.release()

	

main()

