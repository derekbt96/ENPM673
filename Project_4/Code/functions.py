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


	def apply2(self,img):
		if self.start_bounds is None:
			self.get_start_bound(gray)
			print(self.bounds)
		else:
			self.bounds = self.start_bounds
			
		I_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		return I_x, self.bounds
		


	def align(self, T, I, rect, dp0=np.zeros(6), threshold=0.001, iterations=50):

		cap = get_frames(self.vid)
		T_rows, T_cols = T.shape
		I_rows, I_cols = I.shape
		dp = dp0

		for i in range (iterations):
			print(i)
			# Forward warp matrix from frame_t to frame_t+1
			W = np.float32([ 
				[1+dp[0], dp[2], dp[4]], 
				[dp[1], 1+dp[3], dp[5]] ])

			# Warp image from frame_t+1 to frame_t and crop it
			I_warped = cv2.warpAffine(I, cv2.invertAffineTransform(W), (I_cols, I_rows))
			# print(np.shape(I_warped))
			# print(T_rows)
			# print(T_cols)
			I_warped = cap.crop_im(I_warped, rect)
			
			# Image gradients
			dI_x = cv2.Sobel(I_warped, cv2.CV_64F, 1, 0, ksize=3)
			dI_y = cv2.Sobel(I_warped, cv2.CV_64F, 0, 1, ksize=3)

			dI = np.dstack((np.tile(dI_x.flatten(), (6, 1)).T, np.tile(dI_y.flatten(), (6, 1)).T))
			dI = np.reshape(dI, (T_rows*T_cols, 2, 6))

			dW = []
			for y in range(rect[1], rect[3], 1):
				for x in range(rect[0], rect[2], 1):
					dW.append(np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]]))

			# Steepest descent
			A = np.sum(np.sum(np.multiply(dI, dW), axis=1), axis=0).reshape(1,6)
			# Hessian 
			H = np.matmul(A.T, A)
			# Error image 
			err_im = (T - I_warped).flatten()
			err_im = np.reshape(err_im, (1, len(err_im)))

			del_p = np.sum(np.matmul(np.linalg.inv(H), np.matmul(A.T, err_im)), axis=1)

			# Test for convergence and exit 
			if np.linalg.norm(del_p) <= threshold: 
				break

			# Update the parameters
			dp = dp + del_p

		return dp





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


		grad_x = -self.last_grad_x[self.bounds[1]:self.bounds[3],self.bounds[0]:self.bounds[2]]
		grad_y = -self.last_grad_y[self.bounds[1]:self.bounds[3],self.bounds[0]:self.bounds[2]]
		# cv2.imshow('T',np.hstack([grad_x, grad_y]))
		# cv2.waitKey(1)

		x = np.linspace(0, grad_x.shape[0]-1, grad_x.shape[0])
		y = np.linspace(0, grad_y.shape[1]-1, grad_y.shape[1])
		x1, y1 = np.meshgrid(y, x)



		# print(x1.shape)
		# print(y1.shape)
		# print(grad_x.shape)
		# print(grad_x.shape)
		
		d_I_Wxp = [np.multiply(x1, grad_x), np.multiply(x1, grad_y), np.multiply(y1, grad_x),np.multiply(y1, grad_y), grad_x, grad_y]
		

		H = np.array([[np.sum(np.multiply(d_I_Wxp[i], d_I_Wxp[j])) for i in range(6)] for j in range(6)])	
		Hinv = np.linalg.pinv(H)


		bad_itr = 0
		min_cost = -1
		minW = np.matrix([[1., 0., 0.], [0., 1., 0.]])
		W = np.matrix([[1., 0., 0.], [0., 1., 0.]])


		for iterantion in range(150):
			# print(W)
			warped = cv2.warpAffine(self.last_frame.copy(),W,(self.shape[0],self.shape[1]))
			I = warped[self.bounds[1]:self.bounds[3],self.bounds[0]:self.bounds[2]]

			# cv2.imshow('I',I)
			# cv2.waitKey(1)

			# print('I: ',I.shape)

			error = 3.*np.absolute(np.matrix(T, dtype='int') - np.matrix(I, dtype='int'))
			# cv2.imshow('E',np.matrix(error.copy(),np.uint8))
			# cv2.waitKey(1)
			# print(np.mean(error))

			error_grad = np.matrix([[np.sum(np.multiply(i, error))] for i in d_I_Wxp])
			mean_cost = np.sum(np.absolute(error_grad))
			p = Hinv.dot(error_grad)
			

			dp = np.matrix([[p[0,0],p[2,0],p[4,0]], [p[1,0],p[3,0],p[5,0]]])
			W = W + dp
			

			if (min_cost == -1):
				min_cost = mean_cost
			elif (min_cost >= mean_cost):
				min_cost = mean_cost
				bad_itr = 0
				minW = W
			else:
				bad_itr += 1

			if (bad_itr == 2):
				print('Bad')
				return W
			
			# print(mean_cost)
			# print(np.sum(np.absolute(p)))
			if (np.sum(np.absolute(p)) < 0.0006):
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
