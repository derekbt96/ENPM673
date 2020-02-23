import numpy as np 
import cv2




def main():
	lena = cv2.imread('Lena.png')
	marker = cv2.imread('ref_marker.png')
	marker_grid = cv2.imread('ref_marker_grid.png')


	# decode(marker)

	
	K = np.transpose(np.array([[1406.08415449821,0,0],
	[2.20679787308599, 1417.99930662800,0],
	[1014.13643417416, 566.347754321696,1]]))
	
	
	tag0 = cv2.VideoCapture('Tag1.mp4')
	# tag1 = cv2.VideoCapture('Tag1.mp4')
	# tag2 = cv2.VideoCapture('Tag2.mp4')


	'''
	# Setup SimpleBlobDetector parameters.
	params = cv2.SimpleBlobDetector_Params()
	 
	# Change thresholds
	params.minThreshold = 1;
	params.maxThreshold = 255;
	 
	# Filter by Area.
	params.filterByArea = False
	params.minArea = 15
	 
	# Filter by Circularity
	# params.filterByCircularity = False
	# params.minCircularity = 0.1
	 
	# Filter by Convexity
	# params.filterByConvexity = False
	# params.minConvexity = 0.87
	 
	# Filter by Inertia
	# params.filterByInertia = False
	# params.minInertiaRatio = 0.01
	
	# detector = cv2.SimpleBlobDetector_create(params)
	detector = cv2.SimpleBlobDetector_create()
	'''
	while(tag0.isOpened()):
		ret, frame = tag0.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		ret,th1 = cv2.threshold(gray,225,255,cv2.THRESH_BINARY)



		contours, hierarchy = cv2.findContours(th1 ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contour = contours[0]


		draw = np.zeros(frame.shape,np.uint8)
		# draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
		
		# cv2.drawContours(draw, [contour], 0, (0,255,0), 2)
		cv2.fillPoly(draw, pts = [contour], color=(255,255,255))
		ret,th_e = cv2.threshold(draw,250,255,cv2.THRESH_BINARY)
		kernel = np.ones((5,5),np.uint8)
		eroded_mask = cv2.erode(th_e,None,iterations = 5)#kernel,iterations = 5)


		# Find bounding rectangle
		# rect = cv2.minAreaRect(contour)
		# box = cv2.boxPoints(rect)
		# box = np.int0(box)
		# draw = cv2.drawContours(draw,[box],0,(0,0,255),2)


		
		# gray = np.float32(gray)
		mask = np.zeros(th1.shape,np.uint8)
		dst = cv2.cornerHarris(th1,9,9,0.08)
		mask[dst>0.01*dst.max()]=255
		# print(dst.shape)
		result = cv2.bitwise_and(eroded_mask, eroded_mask, mask = mask)
		
		
		
		# contours, hierarchy = cv2.findContours(th1 ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# contour = contours[0]
		# draw = np.zeros(frame.shape,np.uint8)
		# result = cv2.drawContours(result,contours,0,(0,0,255),2)


		# leftmost = tuple(result[result[:,:,0].argmin()][0])
		# rightmost = tuple(result[result[:,:,0].argmax()][0])
		# topmost = tuple(result[result[:,:,1].argmin()][0])
		# bottommost = tuple(result[result[:,:,1].argmax()][0])
		

		# result = cv2.circle(result, leftmost, 15, (255, 0, 0) , 5) 
		# result = cv2.circle(result, rightmost, 15, (255, 0, 0) , 5) 
		# result = cv2.circle(result, topmost, 15, (255, 0, 0) , 5) 
		# result = cv2.circle(result, bottommost, 15, (255, 0, 0) , 5) 

		# epsilon = 0.1*cv.arcLength(cnt,True)
		# approx = cv.approxPolyDP(cnt,epsilon,True)

		# cv2.drawContours(th1,contours,0,(0,0,255),2)


		
		# keypoints = detector.detect(gray)
		# print(keypoints)
		# Draw detected blobs as red circles.
		# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
		# im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		


		# gray = np.float32(gray)
		# dst = cv2.cornerHarris(th1,9,9,0.04)
		# print(dst.shape)
		# frame[dst>0.01*dst.max()]=[0,0,255]

		frameR = cv2.resize(result, (800, 450), interpolation = cv2.INTER_AREA)
		# frameR = cv2.resize(th1, (800, 450), interpolation = cv2.INTER_AREA)
		# while True:
		cv2.imshow('frame',frameR)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		# break

	tag0.release()
	# tag1.release()
	# tag2.release()
	cv2.destroyAllWindows()

	
	'''
	tag0 = cv2.VideoCapture('Tag0.mp4')
	ret, frame = tag0.read()
	tag0.release()



	x = np.array([1, 512, 512, 1])
	y = np.array([1, 1, 512, 512])
	xp = np.array([920, 1130, 1310, 1080])
	yp = np.array([485, 405, 650, 760])


	H = find_homography(x,y,xp,yp)
	added = superImpose(H,lena,frame)

	while True:
		cv2.imshow('frame',blank_image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	'''

	'''
	# frame = cv2.circle(frame, (1080,760), 15, (255, 0, 0) , 5) 
	
	gray = cv2.cvtColor(marker,cv2.COLOR_BGR2GRAY)

	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)

	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst,None)

	# Threshold for an optimal value, it may vary depending on the image.
	marker[dst>0.01*dst.max()]=[0,0,255]
	
	
	# cv2.imshow('dst',added)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()
	'''
	


def correct_holes(img,msk):

	pass


def tag_image(img,pnt,tag):
	
	font = cv2.FONT_HERSHEY_SIMPLEX
	# org 
	org = (50, 50) 
	# fontScale 
	fontScale = 1
	# Blue color in BGR 
	color = (255, 0, 0) 
	# Line thickness of 2 px 
	thickness = 2
	# Using cv2.putText() method 
	image = cv2.putText(img, str(tag), pnt, font, fontScale, color, thickness, cv2.LINE_AA) 
	image = cv2.circle(image, pnt, thickness, color, thickness)
	return img


def decode(marker_img):

	scale = 10
	marker_img = cv2.resize(marker_img, (scale*8, scale*8), interpolation = cv2.INTER_AREA)
	marker_img = cv2.cvtColor(marker_img,cv2.COLOR_BGR2GRAY)
	marker_img = cv2.GaussianBlur(marker_img,(3,3),0)
	ret,th1 = cv2.threshold(marker_img,200,255,cv2.THRESH_BINARY)
	
	orientation_tags = np.array([th1[int(scale*2.5),int(scale*2.5)],
						th1[int(scale*2.5),int(scale*5.5)],
						th1[int(scale*5.5),int(scale*2.5)],
						th1[int(scale*5.5),int(scale*5.5)]])
	orientation_tags = (orientation_tags > 1).astype(int)

	data_tags =np.array([th1[int(scale*3.5),int(scale*3.5)],
						th1[int(scale*3.5),int(scale*4.5)],
						th1[int(scale*4.5),int(scale*3.5)],
						th1[int(scale*4.5),int(scale*4.5)]])
	data_tags = (data_tags > 1).astype(int)

	corner = np.argmax(orientation_tags)
	if corner == 3:
		tag_ID = data_tags
	elif corner == 2:
		tag_ID = np.array([data_tags[1],data_tags[3],data_tags[0],data_tags[2]])
		th1 = np.rot90(th1)
	elif corner == 0:
		tag_ID = np.array([data_tags[3],data_tags[2],data_tags[1],data_tags[0]])
		th1 = np.rot90(th1)
		th1 = np.rot90(th1)
	elif corner == 1:
		tag_ID = np.array([data_tags[2],data_tags[0],data_tags[3],data_tags[1]])
		th1 = np.rot90(th1)
		th1 = np.rot90(th1)
		th1 = np.rot90(th1)

	# while True:
	# 	cv2.imshow('frame',th1)
	# 	if cv2.waitKey(1) & 0xFF == ord('q'):
	# 		break
	
	print(tag_ID)
	return tag_ID


def superImpose(H,src,dest):

	# blank_image = np.zeros(dest.shape, np.uint8)
	# image_mask = np.zeros((dest.shape[0],dest.shape[1],1), np.uint8)

	
	# for col in range(src.shape[0]):
	# 	for row in range(src.shape[1]):
	for col in range(50):
		for row in range(50):
			xy_new = np.dot(H,np.array([col,row,1]))
			xy_new = xy_new[0,0:2]/xy_new[0,2]
			
			dest[int(xy_new[0,1]),int(xy_new[0,0])] = src[col,row]

	# for col in range(src.shape[0]):
		# for row in range(src.shape[1]):
	for col in range(50):
		for row in range(50):
			xy_new = np.dot(H,np.array([col,row,.5]))
			xy_new = xy_new[0,0:2]/xy_new[0,2]
			
			dest[int(xy_new[0,1]),int(xy_new[0,0])] = src[col,row]
	
	# warp = cv2.warpPerspective(src, H, (dest.shape[1],dest.shape[0]))
	# indx = np.where(warp != [0,0,0])
	# dest[indx] = warp[indx]


	destR = cv2.resize(dest, (800, 450), interpolation = cv2.INTER_AREA)
	while True:
		cv2.imshow('frame',dest)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	return dest


def find_homography(x,y,xp,yp):
	A = np.matrix([ [-x[0], -y[0], -1, 0, 0, 0, x[0]*xp[0], y[0]*xp[0], xp[0]],
					[0, 0, 0, -x[0], -y[0], -1, x[0]*yp[0], y[0]*yp[0], yp[0]],
					[-x[1], -y[1], -1, 0, 0, 0, x[1]*xp[1], y[1]*xp[1], xp[1]],
					[0, 0, 0, -x[1], -y[1], -1, x[1]*yp[1], y[1]*yp[1], yp[1]],
					[-x[2], -y[2], -1, 0, 0, 0, x[2]*xp[2], y[2]*xp[2], xp[2]],
					[0, 0, 0, -x[2], -y[2], -1, x[2]*yp[2], y[2]*yp[2], yp[2]],
					[-x[3], -y[3], -1, 0, 0, 0, x[3]*xp[3], y[3]*xp[3], xp[3]],
					[0, 0, 0, -x[3], -y[3], -1, x[3]*yp[3], y[3]*yp[3], yp[3]] ])

	evl_AtA, V = np.linalg.eig(np.dot(np.transpose(A),A))

	H = np.reshape(V[:,-1],[3,3])

	# Z = np.dot(A,H)
	# print(Z)
	# print(Hs)

	return H




main()