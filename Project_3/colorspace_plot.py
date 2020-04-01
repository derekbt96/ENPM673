import numpy as np 
import cv2
from matplotlib import pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d import axes3d, Axes3D  





def main():
	
	capture = cv2.VideoCapture('detectbuoy.avi')
	capture_mask = cv2.VideoCapture('buoy_mask.avi')

	mask_gen = color_mask()


	ret, frame = capture.read()
	mask = mask_gen.get_mask()
	print()

	(mask1, mask2, mask3) = cv2.split(mask)


	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
	color_seg1 = frame.copy()
	color_seg2 = frame.copy()
	color_seg3 = frame.copy()
	color_seg1[mask1 == 0] = (0,0,0)
	color_seg2[mask2 == 0] = (0,0,0)
	color_seg3[mask3 == 0] = (0,0,0)

	color_seg1 = color_seg1.reshape((480*640,3))
	color_seg2 = color_seg2.reshape((480*640,3))
	color_seg3 = color_seg3.reshape((480*640,3))
	
	color_seg1 = color_seg1[~np.all(color_seg1 == 0, axis=1)]
	color_seg2 = color_seg2[~np.all(color_seg2 == 0, axis=1)]
	color_seg3 = color_seg3[~np.all(color_seg3 == 0, axis=1)]
	
	color_segR = frame.copy()
	color_segR[mask1 != 0] = (0,0,0)
	color_segR[mask2 != 0] = (0,0,0)
	color_segR[mask3 != 0] = (0,0,0)
	color_segR = color_segR.reshape((480*640,3))
	color_segR = color_segR[~np.all(color_segR == 0, axis=1)]
	
	fig = plt.figure()
	ax = Axes3D(fig)
	
	print(color_seg1.shape)
	print(color_segR.shape)
	

	ax.scatter(color_seg1[:,0], color_seg1[:,1], color_seg1[:,2], c = (.8,.8,0))
	ax.scatter(color_seg2[:,0], color_seg2[:,1], color_seg2[:,2], c = (1,0,0))
	ax.scatter(color_seg3[:,0], color_seg3[:,1], color_seg3[:,2], c = (0,1,0))
	ax.scatter(color_segR[::30,0], color_segR[::30,1], color_segR[::30,2], c = (.6,.6,.6))
	# ax.scatter(R, G, B, c = np.transpose(col))
	# ax.scatter(R, G, B, c = np.transpose(col))
	# ax.scatter(R, G, B, c = np.transpose(col))

	ax.set_xlabel('H')
	ax.set_ylabel('S')
	ax.set_zlabel('V')
	ax.view_init(elev=70., azim=-150)
	plt.show()

	capture.release()
	mask_gen.cap.release()

	



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