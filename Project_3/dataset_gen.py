import numpy as np 
import cv2
from matplotlib import pyplot as plt
from scipy.io import loadmat

from functions import color_mask, color_data



def main():
	
	capture = cv2.VideoCapture('detectbuoy.avi')
	capture_mask = cv2.VideoCapture('buoy_mask.avi')

	mask_gen = color_mask()


	
	frame_indx = 0
	frame_indx2 = 0


	color1_train = np.zeros((1,3),np.uint8)
	color2_train = np.zeros((1,3),np.uint8)
	color3_train = np.zeros((1,3),np.uint8)

	color1_test = np.zeros((1,3),np.uint8)
	color2_test = np.zeros((1,3),np.uint8)
	color3_test = np.zeros((1,3),np.uint8)

	while(True):

		
		ret, frame = capture.read()
		
		frame_indx += 1
		frame_indx2 += 1
		if frame is None:
			np.save('color_data/training_buoy1_data.npy',color1_train)
			np.save('color_data/training_buoy2_data.npy',color2_train)
			np.save('color_data/training_buoy3_data.npy',color3_train)
			np.save('color_data/testing_buoy1_data.npy',color1_test)
			np.save('color_data/testing_buoy2_data.npy',color2_test)
			np.save('color_data/testing_buoy3_data.npy',color3_test)
			break

		

		mask = mask_gen.get_mask()
		color_seg1, color_seg2, color_seg3 = mask_gen.get_color_arrays(frame,mask)


				
		if frame_indx2 < 5:
			color1_train = np.append(color1_train,color_seg1,axis=0)
			color2_train = np.append(color2_train,color_seg2,axis=0)
			color3_train = np.append(color3_train,color_seg3,axis=0)
		elif frame_indx2 == 5:
			color1_test = np.append(color1_test,color_seg1,axis=0)
			color2_test = np.append(color2_test,color_seg2,axis=0)
			color3_test = np.append(color3_test,color_seg3,axis=0)
		else:
			color1_test = np.append(color1_test,color_seg1,axis=0)
			color2_test = np.append(color2_test,color_seg2,axis=0)
			color3_test = np.append(color3_test,color_seg3,axis=0)
			frame_indx2 = 0



		cv2.imshow('masks',mask)
		
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
	capture.release()
	mask_gen.cap.release()
	cv2.destroyAllWindows()

	


main()

