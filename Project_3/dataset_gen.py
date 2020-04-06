import numpy as np 
import cv2
from matplotlib import pyplot as plt
from scipy.io import loadmat

from functions import color_mask, color_data



def main():
	
	capture = cv2.VideoCapture('detectbuoy.avi')
	# capture_mask = cv2.VideoCapture('buoy_mask.avi')

	mask_gen = color_mask()

	frame_indx = -1


	color1_train = np.array([])
	color2_train = np.array([])
	color3_train = np.array([])

	frame_arr = np.random.randint(0, 199, (20, 1)) 


	while(True):

		
		ret, frame = capture.read()
		
		frame_indx += 1
		if frame is None:
			break

		if frame_indx == 0 or np.equal(np.mod(frame_arr, frame_indx), 0).any():

			mask = mask_gen.get_mask(frame,frame_indx)

			# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			
			color_seg1, color_seg2, color_seg3 = mask_gen.get_color_arrays(frame,mask)


			if frame_indx == 0:
				color1_train = color_seg1
				color2_train = color_seg2
				color3_train = color_seg3
			else:	
				color1_train = np.append(color1_train,color_seg1,axis=0)
				color2_train = np.append(color2_train,color_seg2,axis=0)
				color3_train = np.append(color3_train,color_seg3,axis=0)
			


		cv2.imshow('masks',mask)
		
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
	capture.release()
	mask_gen.cap.release()
	cv2.destroyAllWindows()

	np.save('color_data/training_buoy1_data.npy',color1_train)
	np.save('color_data/training_buoy2_data.npy',color2_train)
	np.save('color_data/training_buoy3_data.npy',color3_train)
	# np.save('color_data/training_buoy1_data_HSV.npy',color1_train)
	# np.save('color_data/training_buoy2_data_HSV.npy',color2_train)
	# np.save('color_data/training_buoy3_data_HSV.npy',color3_train)



main()

