import numpy as np 

def fundamental(points_f1, points_f2):

	A1 = np.multiply(points_f1, np.tile(points_f2[:,0],(3,1)).T)
	A2 = np.multiply(points_f1, np.tile(points_f2[:,1],(3,1)).T)
	A3 = points_f1
	A = np.hstack((np.hstack((A1, A2)), A3))

	U, S, V = np.linalg.svd(A)
	F = np.reshape(V[:,-1], (3,3))
	
	U, S, V = np.linalg.svd(F)
	S = np.diag(S)
	# enforce rank 2 condition
	S[2,2] = 0
	# recalculate Fundamental matrix
	F = np.matmul(np.matmul(U, S), np.transpose(V)) 

	return F

def normalized_fundamental(points_f1, points_f2):

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
	s_f1 = np.sqrt(np.var(f1_centred, axis=0)*(l/(l-1)))
	s_f2 = np.sqrt(np.var(f2_centred, axis=0)*(l/(l-1)))
	# Transformation matrix
	Ta = np.matmul([
		[1/s_f1[0], 0, 0], 
		[0, 1/s_f1[1], 0], 
		[0, 0, 1]],

		[[1, 0, -centroid_f1[0]], 
		[0, 1, -centroid_f1[1]], 
		[0, 0, 1]])
	
	Tb = np.matmul([
		[1/s_f2[0], 0, 0], 
		[0, 1/s_f2[1], 0], 
		[0, 0, 1]],

		[[1, 0, -centroid_f2[0]], 
		[0, 1, -centroid_f2[1]], 
		[0, 0, 1]])
	# Normalized points
	Normalized_f1 = np.matmul(Ta, points_f1.T).T
	Normalized_f2 = np.matmul(Tb, points_f2.T).T

	F_norm = fundamental(Normalized_f1, Normalized_f2)
	nF = np.matmul(np.matmul(np.transpose(Tb), F_norm), Ta)

	return nF

def ransac_fundamental(points_f1, points_f2): 

	l = len(points_f2)
	
	points_f1 = np.reshape(points_f1[:,0], (l,2))
	points_f2 = np.reshape(points_f2[:,0], (l,2))

	# Change to homogeneous coordinates
	points_f1 = np.hstack((points_f1, np.ones((l,1))))
	points_f2 = np.hstack((points_f2, np.ones((l,1))))
	
	# Initialize Fundamental matrix 
 	best_F = np.zeros((3,3))

	# threshold for model convergence 
	thresh = 0.01
	# Number of points selected for a given iteration (8-point algo)
	it_points = 8
	# total number of iterations for which ransac should run
	total_it = 100
 	max_inliers = 0

 	for it in range(total_it):
 		print(it)
 		rand_index = np.random.choice(l, it_points, replace=True)

 		nF = normalized_fundamental(points_f1[rand_index], points_f2[rand_index])
 		
 		epipolar_constraint = np.sum(np.multiply(points_f2, 
 			np.transpose(np.matmul(nF, np.transpose(points_f1)))), 1)

 		current_inliers = len(np.where(abs(epipolar_constraint) < thresh)[0])

 		if (current_inliers > max_inliers):
 			best_F = nF
 			max_inliers = current_inliers

 	error = np.sum(np.multiply(points_f2, 
 			np.transpose(np.matmul(best_F, np.transpose(points_f1)))), 1)
 	indices = np.argsort(abs(error))

 	# Pick out the least erroneous 20 inliers
 	inliers_f1 = points_f1[indices[:20]] 
 	inliers_f2 = points_f2[indices[:20]]

 	return best_F, inliers_f1, inliers_f2
