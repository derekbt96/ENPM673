import numpy as np 
import cv2
from matplotlib import pyplot as plt



class LK_tracker:
	def __init__(self):
		
		self.bounds = []
		self.bad_corners = []


		
	def apply(self,img):

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		if len(self.bounds) == 0:
			
			# self.get_start_bound(gray)
			self.bounds = [71, 51, 191, 152]
			

			self.template = gray[self.bounds[1]:self.bounds[3],self.bounds[0]:self.bounds[2]]
			self.template_size = self.template.shape
			
			self.last_frame = gray
			self.shape = img.shape

			self.corners = cv2.goodFeaturesToTrack(self.template,25,0.01,10)
			self.corners = np.squeeze(np.int0(self.corners))
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

		self.last_frame = gray
		
		out_image = self.label_corners(gray.copy())

		return out_image


	def compute_warp(self,img, indx):

		window = 10

		if indx >= self.corners.shape[0]:
			return

		# pnt = self.corners[indx,:]
		pnt = np.array([self.corners[indx,1],self.corners[indx,0]])

		if pnt[0] + window > self.shape[0] or pnt[1] + window > self.shape[1] or (pnt < 0).any():
			print("out of bounds, deleting "+str(indx))
			print(self.bad_corners)
			print(pnt)
			self.bad_corners.append([pnt[0],pnt[1]])
			print(self.bad_corners)
			
			# self.bad_corners = np.append([self.bad_corners,pnt],axis=0)
			self.corners = np.delete(self.corners, indx, 0)
			return


		x = np.linspace(0, window-1, window)
		y = np.linspace(0, window-1, window)
		x1, y1 = np.meshgrid(x, x)

		T = img[pnt[0]:pnt[0]+window, pnt[1]:pnt[1]+window]


		last_grad_x_template = self.last_grad_x[pnt[0]:pnt[0]+window, pnt[1]:pnt[1]+window]
		last_grad_y_template = self.last_grad_y[pnt[0]:pnt[0]+window, pnt[1]:pnt[1]+window]
		
		delWxp = np.array([np.multiply(x1, last_grad_x_template), np.multiply(x1, last_grad_y_template), np.multiply(y1, last_grad_x_template),np.multiply(y1, last_grad_y_template), last_grad_x_template, last_grad_y_template])
		# print(delWxp.shape)


		H = np.array([[np.sum(np.multiply(delWxp[i], delWxp[j])) for i in range(6)] for j in range(6)])
		Hinv = np.linalg.pinv(H)
		# print(H.shape)

		


		p = np.zeros((6,1))
		p1, p2, p3, p4, p5, p6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
		
		k = 0
		bad_itr = 0
		min_cost = -1
		minW = np.matrix([[1., 0., 0.], [0., 1., 0.]])
		W = np.matrix([[1., 0., 0.], [0., 1., 0.]])

		while (k <= 10):
			k += 1
			warped = cv2.warpAffine(self.last_frame,W,(self.shape[0],self.shape[1]))
			I = warped[pnt[0]:pnt[0]+window, pnt[1]:pnt[1]+window]

			error = np.absolute(np.matrix(I, dtype='int') - np.matrix(T, dtype='int'))
			# print(error.shape)
			

			steepest_error = np.matrix([[np.sum(np.multiply(i, error))] for i in delWxp])
			mean_cost = np.sum(np.absolute(steepest_error))
			p = Hinv.dot(steepest_error)

			# steepest_error = np.matrix([[np.sum(np.multiply(g, error))] for g in gradOriginalP])
			# mean_cost = np.sum(np.absolute(steepest_error))
			# deltap = Hinv.dot(steepest_error)
			# dp = warpInv(deltap)

			dp = np.zeros((6,1))
			# print(dp.shape)
			
			val = (1 + p[0, 0]) * (1 + p[3, 0]) - p[1, 0] * p[2, 0]
			dp[0, 0] = (-p[0, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
			dp[1, 0] = (-p[1, 0]) / val
			dp[2, 0] = (-p[2, 0]) / val
			dp[3, 0] = (-p[3, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
			dp[4, 0] = (-p[4, 0] - p[3, 0] * p[4, 0] + p[2, 0] * p[5, 0]) / val
			dp[5, 0] = (-p[5, 0] - p[0, 0] * p[5, 0] + p[1, 0] * p[4, 0]) / val

			# print(dp)
			# print(dp.reshape((2,3)))
			
			p1 = p1 + dp[0, 0] + p1 * dp[0, 0] + p3 * dp[1, 0]
			p2 = p2 + dp[1, 0] + p2 * dp[0, 0] + p4 * dp[1, 0]
			p3 = p3 + dp[2, 0] + p1 * dp[2, 0] + p3 * dp[3, 0]
			p4 = p4 + dp[3, 0] + p2 * dp[2, 0] + p4 * dp[3, 0]
			p5 = p5 + dp[4, 0] + p1 * dp[4, 0] + p3 * dp[5, 0]
			p6 = p6 + dp[5, 0] + p2 * dp[4, 0] + p4 * dp[5, 0]

			W = np.matrix([[1+p1,p3,p5], [p2,1+p4,p6]])

			if (min_cost == -1):
				min_cost = mean_cost
			elif (min_cost >= mean_cost):
				min_cost = mean_cost
				bad_itr = 0
				minW = W
			else:
				bad_itr += 1
			if (bad_itr == 3):
				W = minW
				temp = W.dot(np.matrix([pnt[0], pnt[1], 1.0]).T)
				self.corners[indx,0] = temp[1]
				self.corners[indx,1] = temp[0]
				return

			# print(np.sum(np.absolute(p)))
			# print(mean_cost)
			
			if (np.sum(np.absolute(p)) < 0.0006):
				temp = W.dot(np.matrix([pnt[0], pnt[1], 1.0]).T)
				self.corners[indx,0] = temp[1]
				self.corners[indx,1] = temp[0]
				return
		

	def label_corners(self,img):
		temp = img.copy()
		for i in range(self.corners.shape[0]):
			temp = cv2.circle(temp,(self.corners[i,1],self.corners[i,0]),3,255,-1)
		for i in range(len(self.bad_corners)):
			temp = cv2.circle(temp,(self.bad_corners[i][1],self.bad_corners[i][0]),3,0,-1)
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
