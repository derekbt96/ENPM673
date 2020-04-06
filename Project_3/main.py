import numpy as np 
import cv2
from matplotlib import pyplot as plt
from functions import color_mask, color_data, GMM
import scipy
from scipy import io 


def main():
	
	
	
	capture = cv2.VideoCapture('detectbuoy.avi')
	mask_gen = color_mask()

	ret, frame = capture.read()
	mask = mask_gen.get_mask(frame,0)
	
	HSV = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
	color_seg1, color_seg2, color_seg3, color_segR = mask_gen.get_all_arrays(HSV,mask)


	data = color_data()
	
	# gmm_1 = GMM(color_seg1, 1, 100, .25)
	# gmm_1 = GMM(data.train1, 1, 100, .25)
	# gmm_1.train()
	# gmm_1.save_params('color_data/buoy1_')
	# gmm_1.load_params('color_data/buoy1_')
	# mask = gmm_1.apply_gmm(frame)

	# gmm_2 = GMM(data.train2, 4, 100, .25)
	# gmm_2.train()
	# gmm_2.save_params('color_data/buoy2_')
	# gmm_2.load_params('color_data/buoy2_')
	# mask = gmm_2.apply_gmm(HSV)

	# gmm_3 = GMM(color_seg3, 4, 200, .5)
	gmm_3 = GMM(data.train3, 4, 200, .5)
	# gmm_3.train()
	# gmm_3.save_params('color_data/buoy3_')
	gmm_3.load_params('color_data/buoy3_')
	gmm_3.threshold = .85
	mask = gmm_3.apply_gmm(HSV)
	

	cv2.imshow('result',mask)
	cv2.waitKey(-1)

	# out = cv2.VideoWriter('road_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (800,800))
	# frame_indx = -1
	# while(True):
	# 	frame_indx += 1
	# 	ret, frame = capture.read()
	# 	if frame is None:
	# 		break
		
		
	# 	result = masker.get_mask(frame,frame_indx)

		
	# 	# result = cv2.resize(frame, (800, 400), interpolation = cv2.INTER_AREA)
	# 	# out.write(result2)
	# 	cv2.imshow('result',result)
	# 	# break
	# 	# if frame_indx > 50:
	# 		# break
	# 	if cv2.waitKey(50) & 0xFF == ord('q'):
	# 		break

	# # out.release()
	# capture.release()
	# cv2.destroyAllWindows()

main()
