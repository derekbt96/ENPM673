import numpy as np 
import cv2
from matplotlib import pyplot as plt
from functions import color_mask, color_data, GMM, buoy_detector
import scipy
from scipy import io 


# Change training here
Train = False





save_data = False
def main():
	
	
	
	capture = cv2.VideoCapture('detectbuoy.avi')
	data = color_data()
	buoy_detect = buoy_detector()
	
	
	gmm_1 = GMM(data.train1, 1, 100, .3)
	gmm_2 = GMM(data.train2, 4, 100, .25)
	gmm_3 = GMM(data.train3, 4, 200, .8)


	if Train:
		gmm_1.train()
		gmm_2.train()
		gmm_3.train()
	else:
		gmm_1.load_params('color_data/buoy1_')
		gmm_2.load_params('color_data/buoy2_')
		gmm_3.load_params('color_data/buoy3_')
	
	
	gmm_1.save_params('color_data/buoy1_')
	gmm_2.save_params('color_data/buoy2_')
	gmm_3.save_params('color_data/buoy3_')
	


	out = cv2.VideoWriter('GMM_marked.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,480))
	frame_indx = -1
	while(True):
		frame_indx += 1
		ret, frame = capture.read()
		if frame is None:
			break
				
		HSV = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)

		mask1 = gmm_1.apply_gmm(HSV)
		mask2 = gmm_2.apply_gmm(HSV)
		mask3 = gmm_3.apply_gmm(HSV)
		
		# result = cv2.merge([mask1,mask2,mask3])
		
		pnt1,result1 = buoy_detect.detect(mask1)
		pnt2,result2 = buoy_detect.detect(mask2)
		pnt3,result3 = buoy_detect.detect(mask3)
		
		result = cv2.merge([result1,result2,result3])
		
		result = buoy_detect.add_points(result,[pnt1,pnt2,pnt3])

		cv2.imwrite('gmm_marked_points.png',result)
		out.write(result)
		cv2.imshow('result',result)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	out.release()
	capture.release()
	cv2.destroyAllWindows()

main()
