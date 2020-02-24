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
	
	lena = cv2.imread('Lena.png')

	tag0 = cv2.VideoCapture('Tag1.mp4')
	# tag1 = cv2.VideoCapture('Tag1.mp4')
	# tag2 = cv2.VideoCapture('Tag2.mp4')


	while(tag0.isOpened()):
		ret, frame = tag0.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		ret,th1 = cv2.threshold(gray,225,255,cv2.THRESH_BINARY)
		edge = cv2.Canny(th1, 75, 200)


		contours, hierarchy = cv2.findContours(edge ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		area_bounds = [3000, 20000]
		final_contours = []
		final_contour_areas = []
		
		for contour in contours:
			area = cv2.contourArea(contour)
			# print(area)
			arc_L = cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, arc_L/50.0, True)
			# print(approx.shape)
			if area > area_bounds[0] and area < area_bounds[1] and approx.shape[0] == 4: 
				final_contours.append(approx)


		result = tag_image(frame,final_contours)

		# draw = np.zeros(frame.shape,np.uint8)
		# draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
		
		# result = cv2.drawContours(frame,final_contours,-1,(0,0,255),2)


		result = cv2.resize(result, (800, 450), interpolation = cv2.INTER_AREA)

		cv2.imshow('frame',result)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		# break
	tag0.release()
	# tag1.release()
	# tag2.release()
	cv2.destroyAllWindows()

	
	
	
	


def correct_holes(img,msk):

	pass


def tag_image(img,contours):
	
	tags = []
	for contour in contours:
		tag_img = pull_tag(img,contour)

		# cv2.imshow('frame',tag_img)
		# cv2.waitKey(-1)

		tags.append(decode(tag_img))
		
	added_frame = add_tag_imgs(img, contours, tags)
	return added_frame


def pull_tag(img,contour_tag):
	draw = np.zeros([100,100,3],np.uint8)
	x = np.array([1, 100, 100, 1])
	y = np.array([1, 1, 100, 100])
	xp = np.array([contour_tag[0,0,0], contour_tag[1,0,0], contour_tag[2,0,0], contour_tag[3,0,0]])
	yp = np.array([contour_tag[0,0,1], contour_tag[1,0,1], contour_tag[2,0,1], contour_tag[3,0,1]])
	H = find_homography(x,y,xp,yp)
	for col in range(100):
		for row in range(100):
			xy_new = np.dot(H,np.array([col,row,1]))
			xy_new = xy_new[0,0:2]/xy_new[0,2]
			draw[col,row] = img[int(xy_new[0,1]),int(xy_new[0,0])]
	return draw


def decode(marker_image):
	scale = 10
	marker_img = cv2.resize(marker_image, (scale*8, scale*8), interpolation = cv2.INTER_AREA)
	# marker_img2 = cv2.resize(marker_image, (scale*8, scale*8), interpolation = cv2.INTER_AREA)
	marker_img = cv2.cvtColor(marker_img,cv2.COLOR_BGR2GRAY)
	marker_img = cv2.GaussianBlur(marker_img,(3,3),0)
	ret,th1 = cv2.threshold(marker_img,200,255,cv2.THRESH_BINARY)
	
	# kernel = np.ones((3,3),np.uint8)
	# th1 = cv2.erode(th1,kernel,iterations = 1)

	# cv2.imshow('frame',th1)
	# cv2.waitKey(-1)

	orientation_tags = np.array([th1[int(scale*2.3),int(scale*2.3)],
						th1[int(scale*2.3),int(scale*5.7)],
						th1[int(scale*5.7),int(scale*2.3)],
						th1[int(scale*5.7),int(scale*5.7)]])
	orientation_tags = (orientation_tags > 1).astype(int)

	data_tags =np.array([th1[int(scale*3.4),int(scale*3.4)],
						th1[int(scale*3.4),int(scale*4.6)],
						th1[int(scale*4.6),int(scale*3.4)],
						th1[int(scale*4.6),int(scale*4.6)]])
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
	

	# image = marker_img
	# image = cv2.circle(image, (int(scale*2.1),int(scale*5.9)), 2, (255, 0, 0) , 2) 
	# image = cv2.circle(image, (int(scale*5.9),int(scale*5.9)), 2, (255, 0, 0) , 2) 
	# image = cv2.circle(image, (int(scale*2.1),int(scale*2.1)), 2, (255, 0, 0) , 2) 
	# image = cv2.circle(image, (int(scale*5.9),int(scale*2.1)), 2, (255, 0, 0) , 2) 
	# image = cv2.circle(image, (int(scale*3.4),int(scale*4.6)), 2, (255, 0, 0) , 2) 
	# image = cv2.circle(image, (int(scale*4.6),int(scale*4.6)), 2, (255, 0, 0) , 2) 
	# image = cv2.circle(image, (int(scale*4.6),int(scale*3.4)), 2, (255, 0, 0) , 2) 
	# image = cv2.circle(image, (int(scale*3.4),int(scale*3.4)), 2, (255, 0, 0) , 2) 
	# cv2.imshow('frame',image)
	# cv2.waitKey(-1)

	print(tag_ID)
	return tag_ID


def add_tag_imgs(img, contours, tags):

	font = cv2.FONT_HERSHEY_SIMPLEX
	fontScale = 1
	color = (255, 0, 0) 
	thickness = 2
		
	indx = 0
	for contour in contours:
		pnt = (max(contour[:,0,0]),min(contour[:,0,1]))
		image = cv2.putText(img, str(tags[indx]), pnt, font, fontScale, color, thickness, cv2.LINE_AA) 
		indx += 1
	return img


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