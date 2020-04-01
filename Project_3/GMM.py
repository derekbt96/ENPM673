import numpy as np 
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D  




def main():
	data = color_data()
	
	print(np.mean(data.train1,axis=0))
	print(np.mean(data.train2,axis=0))
	print(np.mean(data.train3,axis=0))
	print(np.std(data.train1,axis=0))
	print(np.std(data.train2,axis=0))
	print(np.std(data.train3,axis=0))


	data.plot_data(None)



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

main()