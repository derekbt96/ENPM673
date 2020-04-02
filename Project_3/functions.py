import numpy as np 
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D  



class color_data:
	def __init__(self):
		if False:
			self.train1 = np.load('color_data/training_buoy1_data.npy')
			self.train2 = np.load('color_data/training_buoy2_data.npy')
			self.train3 = np.load('color_data/training_buoy3_data.npy')
			self.test1 = np.load('color_data/testing_buoy1_data.npy')
			self.test2 = np.load('color_data/testing_buoy2_data.npy')
			self.test3 = np.load('color_data/testing_buoy3_data.npy')
		else:
			self.train1 = np.load('color_data/training_buoy1_data_HSV.npy')
			self.train2 = np.load('color_data/training_buoy2_data_HSV.npy')
			self.train3 = np.load('color_data/training_buoy3_data_HSV.npy')
			self.test1 = np.load('color_data/testing_buoy1_data_HSV.npy')
			self.test2 = np.load('color_data/testing_buoy2_data_HSV.npy')
			self.test3 = np.load('color_data/testing_buoy3_data_HSV.npy')

	
	def plot_data(self,color_data):

		if color_data is None:
			B = self.test3[:,0]
			G = self.test3[:,1]
			R = self.test3[:,2]
		else:
			B = color_data[:,0]
			G = color_data[:,1]
			R = color_data[:,2]
		
		# Make a 3D ScaTTER Of RGB values
		fig = plt.figure()
		ax = Axes3D(fig)
		
		col = np.vstack([R, G, B])/255.0
		ax.scatter(R, G, B, c = np.transpose(col))

		ax.set_xlabel('B')
		ax.set_ylabel('G')
		ax.set_zlabel('R')

		plt.show()


class color_mask:
	def __init__(self):
		self.cap = cv2.VideoCapture('buoy_mask.avi')
		self.thres = [127,255]
		self.kernel3 = np.ones((3,3),np.uint8)
		self.kernel5 = np.ones((9,9),np.uint8)
		

	
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


	def get_color_arrays(self,img,masks):

		(mask1, mask2, mask3) = cv2.split(masks)

		# img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		color_seg1 = img.copy()
		color_seg2 = img.copy()
		color_seg3 = img.copy()
		color_seg1[mask1 == 0] = (0,0,0)
		color_seg2[mask2 == 0] = (0,0,0)
		color_seg3[mask3 == 0] = (0,0,0)

		color_seg1 = color_seg1.reshape((480*640,3))
		color_seg2 = color_seg2.reshape((480*640,3))
		color_seg3 = color_seg3.reshape((480*640,3))
		
		color_seg1 = color_seg1[~np.all(color_seg1 == 0, axis=1)]
		color_seg2 = color_seg2[~np.all(color_seg2 == 0, axis=1)]
		color_seg3 = color_seg3[~np.all(color_seg3 == 0, axis=1)]
		
		return color_seg1, color_seg2, color_seg3

	def get_all_arrays(self,img,masks):

		(mask1, mask2, mask3) = cv2.split(masks)

		# img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		color_seg1 = img.copy()
		color_seg2 = img.copy()
		color_seg3 = img.copy()
		color_segR = img.copy()
		
		color_seg1[mask1 == 0] = (0,0,0)
		color_seg2[mask2 == 0] = (0,0,0)
		color_seg3[mask3 == 0] = (0,0,0)

		color_segR[mask1 != 0] = (0,0,0)
		color_segR[mask2 != 0] = (0,0,0)
		color_segR[mask3 != 0] = (0,0,0)
		
		color_seg1 = color_seg1.reshape((480*640,3))
		color_seg2 = color_seg2.reshape((480*640,3))
		color_seg3 = color_seg3.reshape((480*640,3))
		color_segR = color_segR.reshape((480*640,3))
		
		color_seg1 = color_seg1[~np.all(color_seg1 == 0, axis=1)]
		color_seg2 = color_seg2[~np.all(color_seg2 == 0, axis=1)]
		color_seg3 = color_seg3[~np.all(color_seg3 == 0, axis=1)]
		color_segR = color_segR[~np.all(color_segR == 0, axis=1)]

		return color_seg1, color_seg2, color_seg3, color_segR 
