import numpy as np 
import cv2
from matplotlib import pyplot as plt



class LK_tracker:
	def __init__(self,problem):
		
		self.bounds = []
		self.problem = problem
		
		
		if self.problem == 1:
			self.sobel_size_x = 3
			self.sobel_size_y = 3
		elif self.problem == 2:
			self.sobel_size_x = 3
			self.sobel_size_y = 3	
		elif self.problem == 3:
			self.sobel_size_x = 5
			self.sobel_size_y = 5	
			



		if True:
			if self.problem == 1:
				self.start_bounds = [90, 86, 166, 119]
			elif self.problem == 2:
				self.start_bounds = [330, 163, 370, 220]	
			elif self.problem == 3:
				self.start_bounds = [101, 121, 241, 283]	
		else:
			self.start_bounds = None


	def transform_bounds(self,W):
		# print(W)
		temp = np.array([[self.bounds[0], self.bounds[1], 1.0],
						 [self.bounds[0], self.bounds[3], 1.0],
						 [self.bounds[2], self.bounds[1], 1.0],
						 [self.bounds[2], self.bounds[3], 1.0]])
		new_points = W.dot(temp.T)
		self.bounds = [int(np.mean(new_points[0,0:2])),  int(np.mean([new_points[1,0],new_points[1,2]])), int(np.mean(new_points[0,2:4])), int(np.mean([new_points[1,1],new_points[1,3]]))]



	def apply(self,img):

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		if len(self.bounds) == 0:
			

			if self.start_bounds is None:
				self.get_start_bound(gray)
				print(self.bounds)
			else:
				self.bounds = self.start_bounds



			self.template = gray[self.bounds[1]:self.bounds[3],self.bounds[0]:self.bounds[2]]
			self.template_size = self.template.shape
			
			
			
			self.last_frame = gray
			self.shape = np.array([img.shape[1],img.shape[0]])

			self.last_grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=self.sobel_size_x)
			self.last_grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=self.sobel_size_y)
			
			return self.last_frame



		W_new = self.compute_warp(gray)
		
		self.transform_bounds(W_new)

		self.last_grad_x = cv2.Sobel(gray.copy(), cv2.CV_32F, 1, 0, ksize=self.sobel_size_x)
		self.last_grad_y = cv2.Sobel(gray.copy(), cv2.CV_32F, 0, 1, ksize=self.sobel_size_y)

		self.last_frame = gray.copy()
		
		out_image = self.label_corners(gray)

		return out_image


	def compute_warp(self,img):


		T = img[self.bounds[1]:self.bounds[3],self.bounds[0]:self.bounds[2]]
		print(self.bounds)

		grad_x = self.last_grad_x[self.bounds[1]:self.bounds[3],self.bounds[0]:self.bounds[2]]
		grad_y = self.last_grad_y[self.bounds[1]:self.bounds[3],self.bounds[0]:self.bounds[2]]
		# cv2.imshow('T',np.hstack([grad_x, grad_y]))
		# cv2.waitKey(1)
		# cv2.imwrite('T.jpg',T)

		x = np.linspace(0, grad_x.shape[0]-1, grad_x.shape[0])
		y = np.linspace(0, grad_y.shape[1]-1, grad_y.shape[1])
		X, Y = np.meshgrid(y, x)


		d_I_Xdx = np.multiply(X, grad_x)
		d_I_Xdy = np.multiply(X, grad_y)
		d_I_Ydx = np.multiply(Y, grad_x)
		d_I_Ydy = np.multiply(Y, grad_y)

		d_I_Wxp = np.array([d_I_Xdx,d_I_Xdy,d_I_Ydx,d_I_Ydy,grad_x,grad_y])
		# print(d_I_Wxp.shape)

		# d_I_Wxp = [np.multiply(self.x1, last_grad_x_template), np.multiply(self.x1, last_grad_y_template), np.multiply(self.y1, last_grad_x_template),np.multiply(self.y1, last_grad_y_template), last_grad_x_template, last_grad_y_template]
		H = np.array([[np.sum(np.multiply(d_I_Wxp[i], d_I_Wxp[j])) for i in range(6)] for j in range(6)])
		
		# H = np.zeros((6,6))
		# for i in range(6):
			# for j in range(6):
				# H[i,j] = np.sum(np.multiply(d_I_Wxp[i], d_I_Wxp[j]))

		Hinv = np.linalg.pinv(H)


		# bad_itr = 0
		# min_cost = -1
		
		W = np.array([[1., 0., 0.], [0., 1., 0.]])

		for iterantion in range(150):
			# print(W)
			warped = cv2.warpAffine(self.last_frame.copy(),W,(self.shape[0],self.shape[1]))
			I = warped[self.bounds[1]:self.bounds[3],self.bounds[0]:self.bounds[2]]
			
			# cv2.imshow('I',I)
			# cv2.waitKey(-1)


			# print('I: ',I.shape)

			error = np.absolute(np.matrix(T, dtype='int') - np.matrix(I, dtype='int'))
			total_error = np.sum(np.absolute(error))
			
			# cv2.imshow('E',np.matrix(error.copy(),np.uint8))
			# cv2.imwrite('comp.jpg',np.hstack([I,T,np.matrix(error.copy(),np.uint8)]))
			# cv2.waitKey(-1)
			# print(np.mean(error))

			d_I_TI = np.zeros((6,1))
			for i in range(6):
				d_I_TI[i] = np.sum(np.multiply(d_I_Wxp[i], error))
			

			p = Hinv.dot(d_I_TI)
			

			dp = np.matrix([[p[0,0],p[2,0],p[4,0]], [p[1,0],p[3,0],p[5,0]]])
			# dp = np.matrix([[p[0,0],p[1,0],p[2,0]], [p[3,0],p[4,0],p[5,0]]])
			W = W + dp
			

			# if (min_cost == -1):
			# 	min_cost = mean_error
			# elif (min_cost >= mean_error):
			# 	min_cost = mean_error
			# 	bad_itr = 0
			# 	minW = W
			# else:
			# 	bad_itr += 1

			# if (bad_itr == 2):
			# 	print('Bad')
			# 	return W
			
			# print(mean_error)
			# print(np.sum(np.absolute(p)))
			if (np.mean(np.absolute(p)) < .000001):
				print('Good')
				return W
		
		print('Need More iterantions')
		return W


	def label_corners(self,img):
		temp = img.copy()
		# print(self.bounds)
		temp = cv2.circle(temp,(self.bounds[0],self.bounds[1]),3,255,-1)
		temp = cv2.circle(temp,(self.bounds[0],self.bounds[3]),3,255,-1)
		temp = cv2.circle(temp,(self.bounds[2],self.bounds[1]),3,255,-1)
		temp = cv2.circle(temp,(self.bounds[2],self.bounds[3]),3,255,-1)
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
