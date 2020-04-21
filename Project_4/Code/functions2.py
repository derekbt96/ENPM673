import numpy as np 
import cv2
from matplotlib import pyplot as plt



class LK_tracker:
	def __init__(self,problem):
		
		self.bounds = []
		self.bad_corners = []

		self.window = 25

		x = np.linspace(0, self.window-1, self.window)
		y = np.linspace(0, self.window-1, self.window)
		self.x1, self.y1 = np.meshgrid(x, x)

		
	def apply(self,img):

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		if len(self.bounds) == 0:
			
			# self.get_start_bound(gray)
			# self.bounds = [71, 51, 191, 152]
			# self.bounds = [291, 18, 346, 75]
			self.bounds = [85, 74, 167, 122]
			# print(self.bounds)

			self.template = gray[self.bounds[1]:self.bounds[3],self.bounds[0]:self.bounds[2]]
			self.template_size = self.template.shape
			
			self.last_frame = gray
			self.shape = np.array([img.shape[1],img.shape[0]])

			self.corners = cv2.goodFeaturesToTrack(self.template,15,0.01,10)
			self.corners = np.squeeze(np.int0(self.corners))
			# print(self.corners)
			self.corners[:,0] = self.corners[:,0] + self.bounds[0]
			self.corners[:,1] = self.corners[:,1] + self.bounds[1]
			
			# print(self.corners)
			self.last_grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
			self.last_grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
			
			return self.last_frame



		self.grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
		self.grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)


		for indx in range(self.corners.shape[0]):
			self.compute_warp(gray,indx)
		
		self.last_grad_x = self.grad_x
		self.last_grad_y = self.grad_y

		self.last_frame = gray.copy()
		
		out_image = self.label_corners(gray)

		return out_image


	def compute_warp(self,img, indx):



		if indx >= self.corners.shape[0]:
			return


		pnt = self.corners[indx,:]
		# pnt = np.array([self.corners[indx,1],self.corners[indx,0]])

		if pnt[0] + self.window > self.shape[0] or pnt[1] + self.window > self.shape[1] or (pnt < 0).any():
			print("out of bounds, deleting "+str(indx))
			print(pnt[0],self.shape[0],pnt[1],self.shape[1])
			# print(pnt)
			self.bad_corners.append([pnt[0],pnt[1]])
			# self.bad_corners = np.append([self.bad_corners,pnt],axis=0)
			self.corners = np.delete(self.corners, indx, 0)
			return



		T = img[pnt[1]:pnt[1]+self.window, pnt[0]:pnt[0]+self.window]
		# cv2.imshow('T',T)
		# cv2.waitKey(1)

		# print('T: ',T.shape)

		# last_grad_x_template = self.last_grad_x[pnt[1]:pnt[1]+self.window, pnt[0]:pnt[0]+self.window]
		# last_grad_y_template = self.last_grad_y[pnt[1]:pnt[1]+self.window, pnt[0]:pnt[0]+self.window]
		last_grad_x_template = np.matrix([[self.last_grad_x[i, j] for j in range(pnt[1], pnt[1]+self.window)] for i in range(pnt[0], pnt[0]+self.window)])
		last_grad_y_template = np.matrix([[self.last_grad_y[i, j] for j in range(pnt[1], pnt[1]+self.window)] for i in range(pnt[0], pnt[0]+self.window)])
		# cv2.imshow('T',last_grad_x_template)
		# cv2.waitKey(-1)


		d_I_Wxp = [np.multiply(self.x1, last_grad_x_template), np.multiply(self.x1, last_grad_y_template), np.multiply(self.y1, last_grad_x_template),np.multiply(self.y1, last_grad_y_template), last_grad_x_template, last_grad_y_template]
		

		H = np.array([[np.sum(np.multiply(d_I_Wxp[i], d_I_Wxp[j])) for i in range(6)] for j in range(6)])
		Hinv = np.linalg.pinv(H)
		# print(H.shape)

		

		p = np.zeros((2,3))
		p1, p2, p3, p4, p5, p6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
		
		k = 0
		bad_itr = 0
		min_cost = -1
		minW = np.matrix([[1., 0., 0.], [0., 1., 0.]])
		W = np.matrix([[1., 0., 0.], [0., 1., 0.]])


		for iterantion in range(20):
			# k += 1
			# print(W)
			warped = cv2.warpAffine(self.last_frame.copy(),W,(self.shape[0],self.shape[1]))
			I = warped[pnt[1]:pnt[1]+self.window, pnt[0]:pnt[0]+self.window]

			# cv2.imshow('I',I)
			# cv2.waitKey(1)

			# print('I: ',I.shape)

			error = np.absolute(np.matrix(T, dtype='int') - np.matrix(I, dtype='int'))
			# cv2.imshow('E',np.matrix(error.copy(),np.uint8))
			# cv2.waitKey(-1)
			# print(error.shape)
			

			steepest_error = np.matrix([[np.sum(np.multiply(i, error))] for i in d_I_Wxp])
			mean_cost = np.sum(np.absolute(steepest_error))
			p = Hinv.dot(steepest_error)
			

			Wp = np.matrix([[p[0,0],p[2,0],p[4,0]], [p[1,0],p[3,0],p[5,0]]])
			W = W + Wp
			

			# dp = np.zeros((6,1))			
			# val = (1 + p[0, 0]) * (1 + p[3, 0]) - p[1, 0] * p[2, 0]
			# dp[0, 0] = (-p[0, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
			# dp[1, 0] = (-p[1, 0]) / val
			# dp[2, 0] = (-p[2, 0]) / val
			# dp[3, 0] = (-p[3, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
			# dp[4, 0] = (-p[4, 0] - p[3, 0] * p[4, 0] + p[2, 0] * p[5, 0]) / val
			# dp[5, 0] = (-p[5, 0] - p[0, 0] * p[5, 0] + p[1, 0] * p[4, 0]) / val

			# # print(dp)
			# # print(dp.reshape((2,3)))
			

			# # p = p + dp + np.multiply(dp.reshape(2,3),p[0:4].reshape((2,2)))
			
			# p1 = p1 + dp[0, 0] + p1 * dp[0, 0] + p3 * dp[1, 0]
			# p2 = p2 + dp[1, 0] + p2 * dp[0, 0] + p4 * dp[1, 0]
			# p3 = p3 + dp[2, 0] + p1 * dp[2, 0] + p3 * dp[3, 0]
			# p4 = p4 + dp[3, 0] + p2 * dp[2, 0] + p4 * dp[3, 0]
			# p5 = p5 + dp[4, 0] + p1 * dp[4, 0] + p3 * dp[5, 0]
			# p6 = p6 + dp[5, 0] + p2 * dp[4, 0] + p4 * dp[5, 0]

			# W = np.matrix([[1+p1,p3,p5], [p2,1+p4,p6]])

			if (min_cost == -1):
				min_cost = mean_cost
			elif (min_cost >= mean_cost):
				min_cost = mean_cost
				bad_itr = 0
				minW = W
			else:
				bad_itr += 1

			if (bad_itr == 2):
				W = minW
				temp = W.dot(np.matrix([pnt[0], pnt[1], 1.0]).T)
				self.corners[indx,0] = temp[0]
				self.corners[indx,1] = temp[1]
				print(' 	Bad')
				return

			# print(np.sum(np.absolute(p)))
			# print(mean_cost)
			
			if (np.sum(np.absolute(p)) < 0.0006):
				temp = W.dot(np.matrix([pnt[0], pnt[1], 1.0]).T)
				self.corners[indx,0] = temp[0]
				self.corners[indx,1] = temp[1]
				print('Thres')
				return
		

	def label_corners(self,img):
		temp = img.copy()
		for i in range(self.corners.shape[0]):
			temp = cv2.circle(temp,(self.corners[i,0],self.corners[i,1]),3,255,-1)
		for i in range(len(self.bad_corners)):
			temp = cv2.circle(temp,(self.bad_corners[i][0],self.bad_corners[i][1]),3,0,-1)
		return temp


	def bound_callback(self,event,x,y,flags,param):
		# print(event,cv2.EVENT_LBUTTONDBLCLK)
		if event == 4:
			if len(self.bounds) == 0:
				self.bounds = [x,y]
			elif len(self.bounds) == 2:
				self.bounds.extend([x,y])
				# print(self.bounds)


	def get_start_bound(self,img):
		cv2.namedWindow('image')
		cv2.setMouseCallback('image',self.bound_callback)
		cv2.imshow('image',img)

		while len(self.bounds) < 4:
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		cv2.destroyAllWindows()

		

class get_frames:
	def __init__(self,video_seq):
		self.frame_num = 1
		
		self.vid = video_seq
		if video_seq == 1:
			self.file_route = 'Car4/img/'
		elif video_seq == 2:
			self.file_route = 'Bolt/img/'
		else:
			self.file_route = 'DragonBaby/img/'

			
	def get_next_frame(self):
		num = str(self.frame_num)
		num = num.zfill(4)
		
		read_frame = cv2.imread(self.file_route+num+'.jpg')
		self.frame_num = self.frame_num + 1
		return read_frame

	def get_frame(self,indx):
		num = str(indx)
		num = num.zfill(4)
		
		read_frame = cv2.imread(self.file_route+num+'.jpg')

		return read_frame
