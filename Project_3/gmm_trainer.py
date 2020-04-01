import numpy as np 
import cv2
from matplotlib import pyplot as plt
from scipy.io import loadmat





def main():
	
	capture = cv2.VideoCapture('detectbuoy.avi')
	capture_mask = cv2.VideoCapture('buoy_mask.avi')

	mask_gen = color_mask()

	k = 2
	color = "HSV"
	cv2_color = cv2.COLOR_BGR2HSV

	gmm = GMM(
    k, 
    [], [], [], # Loaded from config
    threshold=0.1,
    color=cv2_color
    )
	gmm.load_params(prefix)
	
	frame_indx = 0
	while(True):

		
		ret, frame = capture.read()
		
		frame_indx += 1
		if frame is None:
			break

		

		mask = mask_gen.get_mask()

		color_seg.spin(frame)


		# cv2.imshow('mask1',mask1)
		# cv2.imshow('mask2',mask2)
		# cv2.imshow('mask3',mask3)
		cv2.imshow('result',mask)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
	capture.release()
	mask_gen.cap.release()
	cv2.destroyAllWindows()



class color_mask:
	def __init__(self):
		self.cap = cv2.VideoCapture('buoy_mask.avi')
		self.thres = [127,255]
		self.kernel3 = np.ones((3,3),np.uint8)
		self.kernel5 = np.ones((5,5),np.uint8)

	def get_mask(self):
		ret, frame_mask = self.cap.read()

		if frame_mask is None:
			return None

		ret,thres = cv2.threshold(frame_mask,self.thres[0],self.thres[1],cv2.THRESH_BINARY)
		(mask1, mask2, mask3) = cv2.split(thres)

		mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, self.kernel3)
		mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, self.kernel3)
		mask3 = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, self.kernel3)

		mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, self.kernel5)
		mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, self.kernel5)
		mask3 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, self.kernel5)

		return cv2.merge([mask1,mask2,mask3])


main()



# cd Documents/Documents/Aerospace/ENPM673/ENPM673/Project_3