import numpy as np 
from scipy.spatial.transform import Rotation
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def EstimateFundamentalMatrix(p1, p2):

	
	A = np.matrix([ [p1[0,0]*p2[0,0], 	p1[0,0]*p2[0,1], 	p1[0,0], 	p1[0,1]*p2[0,0], 	p1[0,1]*p2[0,1], 	p1[0,1], 	p2[0,0], 	p2[0,1], 1],
					[p1[1,0]*p2[1,0], 	p1[1,0]*p2[1,1], 	p1[1,0], 	p1[1,1]*p2[1,0], 	p1[1,1]*p2[1,1], 	p1[1,1], 	p2[1,0], 	p2[1,1], 1],
					[p1[2,0]*p2[2,0], 	p1[2,0]*p2[2,1], 	p1[2,0], 	p1[2,1]*p2[2,0], 	p1[2,1]*p2[2,1], 	p1[2,1], 	p2[2,0], 	p2[2,1], 1],
					[p1[3,0]*p2[3,0], 	p1[3,0]*p2[3,1], 	p1[3,0], 	p1[3,1]*p2[3,0], 	p1[3,1]*p2[3,1], 	p1[3,1], 	p2[3,0], 	p2[3,1], 1],
					[p1[4,0]*p2[4,0], 	p1[4,0]*p2[4,1], 	p1[4,0], 	p1[4,1]*p2[4,0], 	p1[4,1]*p2[4,1], 	p1[4,1], 	p2[4,0], 	p2[4,1], 1],
					[p1[5,0]*p2[5,0], 	p1[5,0]*p2[5,1], 	p1[5,0], 	p1[5,1]*p2[5,0], 	p1[5,1]*p2[5,1], 	p1[5,1], 	p2[5,0], 	p2[5,1], 1],
					[p1[6,0]*p2[6,0], 	p1[6,0]*p2[6,1], 	p1[6,0], 	p1[6,1]*p2[6,0], 	p1[6,1]*p2[6,1], 	p1[6,1], 	p2[6,0], 	p2[6,1], 1],
					[p1[7,0]*p2[7,0], 	p1[7,0]*p2[7,1], 	p1[7,0], 	p1[7,1]*p2[7,0], 	p1[7,1]*p2[7,1], 	p1[7,1], 	p2[7,0], 	p2[7,1], 1]])
	# print(A)
	U, S, Vh = np.linalg.svd(A)
	F = np.reshape(Vh[-1,:], (3, 3))
	# F = np.reshape(Vh[:,-1], (3,3))

	U, S, V = np.linalg.svd(F)
	S = np.diag(S)
	# enforce rank 2 condition
	S[2,2] = 0
	# recalculate Fundamental matrix
	F = np.matmul(np.matmul(U, S), np.transpose(V))
	
	return F


def Fundamental(points_f1, points_f2):

	A1 = np.multiply(points_f1, np.tile(points_f1[:,0],(3,1)).T)
	A2 = np.multiply(points_f1, np.tile(points_f1[:,1],(3,1)).T)
	A3 = points_f2
	A = np.hstack([A1, A2, A3])
	# print(A)
	U, S, V = np.linalg.svd(A)
	F = np.reshape(V[:,-1], (3,3))
	
	U, S, V = np.linalg.svd(F)
	S = np.diag(S)
	# enforce rank 2 condition
	S[2,2] = 0
	# recalculate Fundamental matrix
	F = np.matmul(np.matmul(U, S), np.transpose(V))

	return F


def NormalizedFundamental(points_f1, points_f2):

	l = len(points_f2)

	centroid_f1 = np.mean(points_f1, 0)
	centroid_f2 = np.mean(points_f2, 0)
	# Recentre feature points
	f1_centred = points_f1 - np.tile(centroid_f1, (l,1))
	f2_centred = points_f2 - np.tile(centroid_f2, (l,1))
	
	
	"""
	standard deviation. The final multiplication is because 
	python while calculating variance divided it by n and not n-1
	"""
	
	norm_1 = np.mean(np.linalg.norm(f1_centred[:,0:2],axis=1))
	norm_2 = np.mean(np.linalg.norm(f2_centred[:,0:2],axis=1))
	coef_1 = 2**.5 / norm_1
	coef_2 = 2**.5 / norm_1
	

	T1 = np.array([[coef_1, 0, -coef_1*centroid_f1[0]],
				   [0, coef_1, -coef_1*centroid_f1[1]],
				   [0, 0, 1]])

	T2 = np.array([[coef_2, 0, -coef_2*centroid_f2[0]],
				   [0, coef_2, -coef_2*centroid_f2[1]],
				   [0, 0, 1]])

	Normalized_f1 = np.matmul(T1, points_f1.T).T
	Normalized_f2 = np.matmul(T2, points_f2.T).T

	F_norm = EstimateFundamentalMatrix(Normalized_f1, Normalized_f2)
	# print(F_norm)
	# F_norm_og = Fundamental(Normalized_f1, Normalized_f2)
	# print(F_norm_og)
	# print(F_norm - F_norm_og)
	nF = np.matmul(np.matmul(np.transpose(T2), F_norm), T1)
	# print(nF)
	# raise "stop"

	# s_f1 = np.sqrt(np.var(f1_centred, axis=0)*(l/(l-1)))
	# s_f2 = np.sqrt(np.var(f2_centred, axis=0)*(l/(l-1)))
	# # Transformation matrix
	# Ta = np.matmul([
	# 	[1/s_f1[0], 0, 0], 
	# 	[0, 1/s_f1[1], 0], 
	# 	[0, 0, 1]],

	# 	[[1, 0, -centroid_f1[0]], 
	# 	[0, 1, -centroid_f1[1]], 
	# 	[0, 0, 1]])
	
	# Tb = np.matmul([
	# 	[1/s_f2[0], 0, 0], 
	# 	[0, 1/s_f2[1], 0], 
	# 	[0, 0, 1]],

	# 	[[1, 0, -centroid_f2[0]], 
	# 	[0, 1, -centroid_f2[1]], 
	# 	[0, 0, 1]])
	# # Normalized points
	# Normalized_f1 = np.matmul(Ta, points_f1.T).T
	# Normalized_f2 = np.matmul(Tb, points_f2.T).T

	# F_norm = Fundamental(Normalized_f1, Normalized_f2)
	# nF = np.matmul(np.matmul(np.transpose(Tb), F_norm), Ta)

	return nF


def RansacFundamental(points_f1, points_f2): 

	l = len(points_f2)
	
	# points_f1 = np.reshape(points_f1[:,0], (l,2))
	# points_f2 = np.reshape(points_f2[:,0], (l,2))

	# # Change to homogeneous coordinates
	# points_f1 = np.hstack((points_f1, np.ones((l,1))))
	# points_f2 = np.hstack((points_f2, np.ones((l,1))))
	
	# Initialize Fundamental matrix 
	best_F = np.zeros((3,3))

	# threshold for model convergence 
	thresh = .01
	# Number of points selected for a given iteration (8-point algo)
	it_points = 24
	# total number of iterations for which ransac should run
	total_it = 100
	thresh_error = .15
	max_inliers = 0
	mean_error = 1000


	for it in range(total_it):
		# print(it)
		rand_index = np.random.choice(l, it_points, replace=True)

		nF = NormalizedFundamental(points_f1[rand_index], points_f2[rand_index])
		
		epipolar_constraint = np.sum(np.multiply(points_f2, np.transpose(np.matmul(nF, np.transpose(points_f1)))), 1)

		
		current_inliers = len(np.where(abs(epipolar_constraint) < thresh)[0])
		if (current_inliers > max_inliers):
			best_F = nF
			max_inliers = current_inliers

		# temp_error = np.mean(abs(epipolar_constraint))
		# if (temp_error < mean_error):
		# 	best_F = nF
		# 	mean_error = temp_error
		# 	# print(mean_error)
		# 	if mean_error < thresh_error:
		# 		break

	# print(mean_error)
	print(max_inliers,points_f1.shape[0])

	error = np.sum(np.multiply(points_f2, np.transpose(np.matmul(best_F, np.transpose(points_f1)))), 1)
	indices = np.argsort(abs(error))
	# print(np.mean(abs(error)))

	# F,mask = cv2.findFundamentalMat(points_f1,points_f2,cv2.RANSAC, 1,0.999)
	# error = np.sum(np.multiply(points_f2, np.transpose(np.matmul(F, np.transpose(points_f1)))), 1)
	# print(np.mean(abs(error)))
	
	# print(best_F)
	# print(F)

	# print(np.linalg.norm(best_F))
	# print(np.linalg.norm(F))


	# raise 'stop'

	# Pick out the least erroneous k inliers
	k = 30
	inliers_f1 = points_f1[indices[:k]] 
	inliers_f2 = points_f2[indices[:k]]

	# inliers_f1 = points_f1
	# inliers_f2 = points_f2

	# print(np.linalg.det(best_F))
	# print(best_F)


	return best_F, inliers_f1, inliers_f2


def EpipolarLines(img_f1, points_f1, img_f2, points_f2, F):
	h, w, d = np.shape(img_f1)

	top_left = np.array([1,1,1])
	bot_left = np.array([1,h,1])
	top_rt = np.array([w,1,1])
	bot_rt = np.array([w,h,1])

	# Vertical line on the left side of any of the two images 
	line_left = np.cross(top_left, bot_left)
	# Vertical line on the right side of any of the two images
	line_right = np.cross(top_rt, bot_rt)
	
	img_f1 = cv2.UMat(img_f1)
	img_f2 = cv2.UMat(img_f2)

	for it in range(len(points_f1)):
		try:
			# epipolar line in the left image 
			l = np.matmul(F, np.reshape(points_f1[it,:],(3,1)))

			# epipolar line in the right image
			l_ = np.matmul(F.T, np.reshape(points_f2[it,:],(3,1)))

			f1_p1 = np.cross(l.flatten(), line_left)
			f1_p2 = np.cross(l.flatten(), line_right)
			f1_p1 /= f1_p1[2]
			f1_p2 /= f1_p2[2]

			f2_p1 = np.cross(l_.flatten(), line_left)
			f2_p2 = np.cross(l_.flatten(), line_right)
			f2_p1 /= f2_p1[2]
			f2_p2 /= f2_p2[2]

			img_f1 = cv2.line(
				img_f1, 
				(int(f1_p1[0]), int(f1_p1[1])), 
				(int(f1_p2[0]), int(f1_p2[1])), 
				(0,0,255), 
				1) 
			img_f1 = cv2.circle(img_f1, 
				(int(points_f1[it,0]), int(points_f1[it,1])), 
				3, (255,0,0), 2) 

			img_f2 = cv2.line(
				img_f2, 
				(int(f2_p1[0]), int(f2_p1[1])), 
				(int(f2_p2[0]), int(f2_p2[1])), 
				(0,0,255), 
				1) 
			img_f2 = cv2.circle(img_f2, 
				(int(points_f2[it,0]), int(points_f2[it,1])), 
				3, (255,0,0), 2) 
		except:
			pass
	return img_f1, img_f2


def plotCoordinates(pnts):
	plt.figure(figsize=(6,6))
	print(pnts.shape)
	num_set = int(pnts.shape[1]/2)
	print(num_set)
	for i in range(num_set):
		plt.scatter(pnts[:,2*i], pnts[:,2*i+1])#,s=.6)
	plt.grid(True)
	plt.show()


def plotCoordinates3D(pnts):
	plt.figure(figsize=(6,6))
	ax = plt.axes(projection='3d')
	ax.scatter3D(pnts[:,0], pnts[:,1], pnts[:,2])
	ax.scatter3D(0, 0, 0, 'r',)
	
	plt.show()


class camera_pose():
	def __init__(self):
		
		self.pos = np.zeros((3,1))
		self.R = np.identity(3)
		self.X_log = np.zeros((1,3))
		

	def update(self,r,t):
		
		self.pos = self.pos + np.matmul(self.R,np.vstack(t))
		self.R = np.matmul(r,self.R)
		self.X_log = np.vstack([self.X_log, np.hstack(self.pos)])
		
	def update2D(self,r,t):
		rotat = Rotation.from_matrix(r)
		dhdg = rotat.as_euler('yxz')[0]
		# print(dhdg)

		# t = np.array([0,0,1])
		d_pos = np.matmul(self.R,np.vstack(t))
		d_pos[1] = 0

		d_R = np.array([[np.cos(dhdg),0,np.sin(dhdg)],[0,1,0],[-np.sin(dhdg),0,np.cos(dhdg)]])
		
		self.pos = self.pos + d_pos
		self.R = np.matmul(d_R,self.R)
		self.X_log = np.vstack([self.X_log, np.hstack(self.pos)])
		

	def plot3D(self):
		

		plt.figure(figsize=(6,6))
		ax = plt.axes(projection='3d')
		ax.plot3D(self.X_log[:,0], self.X_log[:,1], self.X_log[:,2])
		# plt.plot(self.X_log[:,0], self.X_log[:,2])
		# plt.grid(True)
		plt.show()

	def plot(self):
		
		# print(self.X_log.shape)
		# print(self.X_log)
		# print(self.X_log[:,0])
		
		plt.figure(figsize=(6,6))
		# plt.subplot(211)
		plt.plot(self.X_log[:,2], self.X_log[:,0])
		plt.scatter(self.X_log[:,2], self.X_log[:,0])
		plt.grid(True)
		plt.axis('equal')

		# np.save('logs/straight_opencv.npy',self.X_log)
		# plt.subplot(212)
		# plt.plot(self.X_log[:,2], self.X_log[:,1])
		# plt.scatter(self.X_log[:,2], self.X_log[:,1])
		# plt.grid(True)
		# plt.axis('equal')
		plt.show()

	def save_data(self,name):
		np.save(name,self.X_log)


def recoverPose(F,K,p_old,p_new):

	# Get essential matrix
	# F,mask = cv2.findFundamentalMat(p_old,p_new,cv2.RANSAC, 1,0.999)


	# E = np.matmul(K.T,np.matmul(F,K))
	# U, S, V = np.linalg.svd(E)
	# temp = np.array([[1,0,0],[0,1,0],[0,0,0]])
	# E = np.matmul(np.matmul(U,temp),V)
	# E = E/np.linalg.norm(E)


	
	E,mask = cv2.findEssentialMat(p_old[:,:2], p_new[:,:2],K)

	points, R, t, mask = cv2.recoverPose(E, p_old[:,:2], p_new[:,:2])
	# print(R)
	# print(t)
	return R,t


# Pose from Epipolar geometry
def getCameraPose(F,K,p_old,p_new):



	

	# F,mask = cv2.findFundamentalMat(p_old,p_new,cv2.RANSAC, 1,0.999)

	# Get essential matrix
	E = np.matmul(K.T,np.matmul(F,K))
	# print(np.linalg.matrix_rank(E))
	# U, S, V = np.linalg.svd(E)
	# temp = np.array([[1,0,0],[0,1,0],[0,0,0]])
	# E = np.matmul(U,np.matmul(temp,V))
	# E,mask = cv2.findEssentialMat(p_old[:,:2], p_new[:,:2],K)

	# Get Camera Pose
	W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
	U, S, V = np.linalg.svd(E)
	
	# You can recover the relative pose of the camera
	# between the two frames from the essential matrix
	# up to a scale of t.

	# Upto the scale of t, there are 4 theoretical soln.
	# but only one for which the scene point will be in 
	# in front of the camera: This constraint is termed 
	# as the Cheirality constraint which allows you to 
	# find a unique camera pose 
	# t is (+ or - lambda) U[:,2] 
	
	
	# This is the camera center (translation vector)
	t1 = U[:,2].reshape(3,1)
	t2 = -U[:,2].reshape(3,1)

	# This is the rotation matrix
	R1 = np.matmul(np.matmul(U,W),V)
	R2 = np.matmul(np.matmul(U,W.T),V)


	# To store transform matrices
	T = np.zeros((4,3,4))
	
	# So the 4 candidate poses are:
	T[0,:,:] = np.hstack((R1, t1))
	T[1,:,:] = np.hstack((R1, t2))
	T[2,:,:] = np.hstack((R2, t1))
	T[3,:,:] = np.hstack((R2, t2))

	for i in range(4):
		if (np.linalg.det(T[i,:,:3]) < 0):
			# print(np.linalg.det(T[i,:,:3]))
			T[i,:,:3] = -T[i,:,:3]

	# print(T[1,:,:])

	points, R, t, mask = cv2.recoverPose(E, p_old[:,:2], p_new[:,:2])	
	
	yaw_rec = (Rotation.from_matrix(R)).as_euler('yzx', degrees=True)
	yaw_1 = (Rotation.from_matrix(T[1,:,:3])).as_euler('yzx', degrees=True)
	yaw_2 = (Rotation.from_matrix(T[2,:,:3])).as_euler('yzx', degrees=True)
	# print(yaw_rec)
	# print(yaw_1)
	# print(yaw_2)
	if not (np.allclose(yaw_rec,yaw_1) or np.allclose(yaw_rec,yaw_2)):
		print(R)
		print(T[1,:,:])
		print(T[2,:,:])

	if not (np.allclose(np.hstack(t),T[0,:,3]) or np.allclose(np.hstack(t),T[1,:,3])):
		print(t)
		print(T[0,:,3])
		print(T[1,:,3])

	# print(T)
	# print(R,t)

	return T


def Linear(K, T1, T2, points_f1, points_f2):
	# Triangulation:
	# Linear Solution
	'''
	Generally, the rays joining the camera centres and the 3D world point
	do not intersect due to noise. Therefore triangulation can be solved via
	SVD, finding a least squares solution to a system of equations
	'''
	# Projection Matrix P = KR[I|C]
	P1 = np.matmul( np.matmul(K, T1[:,:3]), 
		np.hstack((np.identity(3), -T1[:,3].reshape(3,1))) )
	P2 = np.matmul( np.matmul(K, T2[:,:3]), 
		np.hstack((np.identity(3), -T2[:,3].reshape(3,1))) )
	
	l = len(points_f1)
	# 3d world point
	X = np.zeros((l,3))

	for i in range(l):
		x, y, z = points_f1[i,:]
		x_, y_, z_ = points_f2[i,:]

		x_f1_mat = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
		x_f2_mat = np.array([[0,-z_,y_],[z_,0,-x_],[-y_,x_,0]])
		
		A = np.vstack((np.matmul(x_f1_mat,P1), np.matmul(x_f2_mat,P2)))
		
		U,S,V = np.linalg.svd(A)

		X_homo = V[:,-1]/V[-1,-1]
		X[i,:] = X_homo[:3].reshape(1,3)
	
	# plotCoordinates3D(X)
	return X


def checkCheirality(T, X):

	max_points = -1000
	R = T[:,:,:3] # 4x3x3
	t = T[:,:,3] # 4x3
	R_final = None
	t_final = None
	X_final = None

	l = len(X[1,:,:])
	# print(X.shape)
	num_points_all = np.zeros((1,4))
	for i in range(4):

		# print("i {}, l {}".format(i,l))
		num_points = 0
		for j in range(l):
			# checkCheirality condition
			if (np.matmul( R[i,2,:].reshape(1,3), (X[i,j,:].reshape(3,1)-t[i,:].reshape(3,1)) ) > 0 and X[i,j,2] >=0):
				num_points += 1
		# 		'''
		# 		# Count the num of points satisfying this condition
		# 		The best camera configuration is the one that produces the maximum number 
		# 		of points satisfying the cheirality condition.
		# 		'''
		
		
		# num_points = np.sum(np.sign(np.matmul(R[i,2,:].reshape(1,3), np.transpose(X[i,:,:]-t[i,:]))))
		# num_points_all[0,i] = num_points
		

		if (num_points > max_points):
			max_points = num_points
			R_final = T[i,:,:3]
			t_final = T[i,:,3]
			X_final = X[i,:,:]
		elif (num_points == max_points and T[i,0,0] > R_final[0,0]):
			R_final = T[i,:,:3]
			t_final = T[i,:,3]
			X_final = X[i,:,:]

			


	yaw = (Rotation.from_matrix(R_final)).as_euler('yxz', degrees=True)
	if (yaw > 25).any():
		print('Yaw: ',yaw,' t: ',t_final)
		print((Rotation.from_matrix(T[0,:,:3])).as_euler('yxz', degrees=True))
		print((Rotation.from_matrix(T[2,:,:3])).as_euler('yxz', degrees=True))
		R_final = np.identity(3)
	# if abs(t_final[0]) > abs(t_final[2]):
	# 	print(T[0,:,3])
	# 	print(T[1,:,3])
	# 	t_final = np.array([0,0,-1])
	# print(t_final)
	
	# print(num_points_all)

	# if t[0,2] > 0:
	# 	t_final = T[0,:,3]
	# elif t[1,2] > 0:
	# 	t_final = T[1,:,3]
	# elif t[2,2] > 0:
	# 	t_final = T[2,:,3]
	# else:
	# 	t_final = T[3,:,3]


	# if R[0,0,0] > 0:
	# 	R_final = T[0,:,:3]
	# elif R[1,0,0] > 0:
	# 	R_final = T[1,:,:3]
	# elif R[2,0,0] > 0:
	# 	R_final = T[2,:,:3]
	# else:
	# 	R_final = T[3,:,:3]

	




	return R_final, t_final, X_final





		


