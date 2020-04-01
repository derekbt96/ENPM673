import numpy as np 
import cv2
from matplotlib import pyplot as plt
from scipy.io import loadmat





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
			np.save('color_data/training_buoy1_data_HSV.npy',color1_train)
			np.save('color_data/training_buoy2_data_HSV.npy',color2_train)
			np.save('color_data/training_buoy3_data_HSV.npy',color3_train)
			np.save('color_data/testing_buoy1_data_HSV.npy',color1_test)
			np.save('color_data/testing_buoy2_data_HSV.npy',color2_test)
			np.save('color_data/testing_buoy3_data_HSV.npy',color3_test)
			break

		

		mask = mask_gen.get_mask()
		(mask1, mask2, mask3) = cv2.split(mask)


		frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		color_seg1 = frame_HSV.copy()
		color_seg2 = frame_HSV.copy()
		color_seg3 = frame_HSV.copy()
		color_seg1[mask1 == 0] = (0,0,0)
		color_seg2[mask2 == 0] = (0,0,0)
		color_seg3[mask3 == 0] = (0,0,0)

		color_seg1 = color_seg1.reshape((480*640,3))
		color_seg2 = color_seg2.reshape((480*640,3))
		color_seg3 = color_seg3.reshape((480*640,3))
		
		color_seg1 = color_seg1[~np.all(color_seg1 == 0, axis=1)]
		color_seg2 = color_seg2[~np.all(color_seg2 == 0, axis=1)]
		color_seg3 = color_seg3[~np.all(color_seg3 == 0, axis=1)]
				
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


		print(color1_train.shape)
		# print(color1_test.shape)

		cv2.imshow('mask1',mask1)
		# cv2.imshow('mask2',mask2)
		# cv2.imshow('mask3',mask3)
		# cv2.imshow('result',color_seg1)

		
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
		self.kernel5 = np.ones((9,9),np.uint8)
		

	def load_data(self):
		temp_train1 = np.load('color_data/training_buoy1_data.npy')
		temp_train2 = np.load('color_data/training_buoy2_data.npy')
		temp_train3 = np.load('color_data/training_buoy3_data.npy')

		temp_test1 = np.load('color_data/testing_buoy1_data.npy')
		temp_test2 = np.load('color_data/testing_buoy2_data.npy')
		temp_test3 = np.load('color_data/testing_buoy3_data.npy')

		return temp_train1, temp_train2, temp_train3, temp_test1, temp_test2, temp_test3


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

		return cv2.merge([mask3,mask2,mask1])


main()



# cd Documents/Documents/Aerospace/ENPM673/ENPM673/Project_3