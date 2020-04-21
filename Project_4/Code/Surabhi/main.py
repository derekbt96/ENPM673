import numpy as np 
import cv2
from matplotlib import pyplot as plt
from functions import LucasKanade, get_frames


# CHANGE THE VARIABLE BELOW TO THE DESIRED OUTPUT PROBLEM
# PROBLEM 1 = Car
# PROBLEM 2 = Bolt
# PROBLEM 3 = Dragon Baby
problem = 1

cap = get_frames(problem)
LK = LucasKanade(problem)
iterations = 50 
threshold = 0.005
# out = cv2.VideoWriter('tracker.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (800,600))

# start_frame = cap.get_next_frame()
# result, rect_ini = LK.apply(start_frame)
# print(rect_ini)
rect_ini = cap.get_bounds()
rect = rect_ini
rects_all = []

# starting
I_x = cap.get_frame()
p = [1, 0, 0, 1, 0, 0]
it = 0
while True:
	print(it)
	it +=1
	see_I_x = cv2.rectangle(cv2.cvtColor(I_x, cv2.COLOR_GRAY2BGR), 
		(int(rect[0]),int(rect[1])), (int(rect[2]), int(rect[3])), (0, 0, 255), 2) 

	T_x = cap.crop_im(I_x, rect)
	I_x1 = cap.get_frame()
	if I_x1 is None:
		break
	dp0 = np.zeros(6)
	dp0[4] = rect[0]
	dp0[5] = rect[1]
	# Warp parameters, solve for dp
	dp = LK.align(T_x, I_x1, rect, p, dp0=dp0,
		threshold=threshold, iterations=iterations)

	p = p + dp
	# Forward warp matrix from frame_t to frame_t+1
	W = np.float32([ 
		[1+p[0], p[2], p[4]], 
		[p[1], 1+p[3], p[5]],
		[0, 0, 1] ])

	rect = np.vstack((rect.reshape(2, 2).T, np.ones(2)))

	new_rect = np.matmul(W, rect).T
	rect = new_rect[:2,:2].flatten()
	rect[0] = int(rect[0])
	rect[1] = int(rect[1])
	rect[2] = int(rect[2])
	rect[3] = int(rect[3])

	rects_all.append(rect)

	I_x = I_x1

	cv2.imshow('result', see_I_x)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break



# # This is just for frame 0 
# frame = cap.get_next_frame()
# if frame is None:
# 	break
# I_x = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 	# grayscale image, template image from frame 1 
# 	# I_x, bounds = tracker.apply(frame)
# T_x = crop_im(I_x, bounds)

# while True:

# 	# Get the image to which you want to match your template
# 	frame = cap.get_next_frame()
# 	if frame is None:
# 		break
# 	frame_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 	# T_x will keep changing for every new image 
# 	T_rows, T_cols = T_x.shape

# 	# dp is what we want to estimate! This is the initial estimate of dp 
# 	dp1, dp2, dp3, dp4, dp5, dp6 = 0, 0, 0, 0, 0, 0
# 	# Warp matrix for the template image
# 	W_x_dp = np.float32([ [1+dp1, dp2, dp3], [dp4, 1+dp5, dp6] ])
# 	# Warp the template image
# 	T_Wx = cv2.warpAffine(T_x, W_x_dp, (T_cols, T_rows))
# 	# and find its x,y gradient 
# 	dT_x = cv2.Sobel(T_Wx, cv2.CV_64F, 1, 0, ksize=3)
# 	dT_y = cv2.Sobel(T_Wx, cv2.CV_64F, 0, 1, ksize=3)

# 	dT = np.dstack((np.tile(dT_x.flatten(), (6, 1)).T, np.tile(dT_y.flatten(), (6, 1)).T))
# 	dT = np.reshape(dT, (T_rows*T_cols,2,6))

# 	dW = []
# 	for y in range(bounds[1], bounds[3], 1):
# 		for x in range(bounds[0], bounds[2], 1):
# 			dW.append(np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]]))

# 	A = np.sum(np.sum(np.multiply(dT, dW), axis=1), axis=0).reshape(1,6)
# 	# Hessian # Precompute~
# 	H = np.matmul(A.T, A)

# 	# run optimization for 50 iterations
# 	# for i in range (iterations):
# 	i = 0
# 	while (i < iterations):

# 		error_img = (crop_im(frame_next, bounds) - T_Wx).flatten()
# 		err = np.reshape(error_img, (1, len(error_img)))
		
# 		del_p = np.sum(np.matmul(np.linalg.inv(H), 
# 			np.matmul(A.T, err)), axis=1)

# 		# Test for convergence and exit 
#         if np.linalg.norm(del_p) <= threshold: 
#         	i = iterations

# 		# Update the warp matrix
#         dM = np.vstack([delta_p.reshape(2, 3) + I, [0, 0, 1]])
#         M = np.matmul(M, np.linalg.inv(dM))

#         i += 1
		
	
	# Applying Affine transform to template image

	# if (cap.frame_num > 1):
	# result = cv2.resize(result, (800, 600), interpolation = cv2.INTER_AREA)
	# out.write(result2)
	# I_x = cv2.cvtColor(I_x, cv2.COLOR_GRAY2BGR)
	# I_x = cv2.rectangle(I_x, (bounds[0],bounds[1]), (bounds[2], bounds[3]), (0, 0, 255), 2) 

	# cv2.imshow('result', I_x)
	# if cv2.waitKey(10) & 0xFF == ord('q'):
	# 	break

# out.release()
cv2.destroyAllWindows()


