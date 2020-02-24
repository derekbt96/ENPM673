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


	lena_mat, marker_mat = gen_mats()
	# temp = marker_mat.reshape(80,80,3)
	# print(temp.shape)
	# print(temp[2,5])
	# tag_capture = cv2.VideoCapture('Tag0.mp4')
	tag_capture = cv2.VideoCapture('Tag1.mp4')
	# tag_capture = cv2.VideoCapture('Tag2.mp4')
	# tag_capture = cv2.VideoCapture('multipleTags.mp4')



	while(tag_capture.isOpened()):

		for a in range(10):
			ret, frame = tag_capture.read()

		ret, frame = tag_capture.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		ret,th1 = cv2.threshold(gray,190,255,cv2.THRESH_BINARY)
		kernel = np.ones((3,3),np.uint8)
		th1 = cv2.erode(th1,kernel,iterations = 1)

		edge = cv2.Canny(th1, 75, 200)



		contours, hierarchy = cv2.findContours(edge.copy() ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		contours = refine_contours(contours)
		
		area_bounds = [3000, 22000]
		final_contours = []
		final_contour_areas = []
		
		for contour in contours:
			area = cv2.contourArea(contour)
			arc_L = cv2.arcLength(contour, True)
			approx = np.squeeze(cv2.approxPolyDP(contour, arc_L/50.0, True))
			if area > area_bounds[0] and area < area_bounds[1] and approx.shape[0] == 4: 
				final_contours.append(approx)

		# if len(final_contours) > 0:
		# 	final_contours = [final_contours[0]]
		# print(final_contours)
		result = tag_image(frame,final_contours,marker_mat.copy())

		# draw = np.zeros(frame.shape,np.uint8)
		# draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
		
		# result = cv2.drawContours(frame,final_contours,-1,(0,0,255),2)


		result = cv2.resize(result, (800, 450), interpolation = cv2.INTER_AREA)

		cv2.imshow('frame',result)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		break

	tag_capture.release()
	cv2.destroyAllWindows()

	
	
	
	
def refine_contours(contours):

	area_bounds = [3000, 30000]
	result_contours = []
	contour_sum = []
	sum_threshold = 100

	for contour in contours:
		area = cv2.contourArea(contour)
		arc_L = cv2.arcLength(contour, True)
		approx = np.squeeze(cv2.approxPolyDP(contour, arc_L/50.0, True))
		
		closeness = (np.abs(contour_sum-np.sum(approx)) < sum_threshold).any()

		if area > area_bounds[0] and area < area_bounds[1] and approx.shape[0] == 4 and not closeness:
			result_contours.append(approx)
			contour_sum.append(np.sum(approx))
			# print(approx)


 #    rect = np.zeros((4, 2), dtype="float32")

	# sum_indx = contour.sum(axis=1)
	# # print(contour.shape)
	# rect[0] = contour[np.argmin(sum_indx)]
	# rect[2] = contour[np.argmax(sum_indx)]

	# diff = np.diff(contour, axis=1)
	# # print(diff)
	# rect[1] = contour[np.argmin(diff)]
	# rect[3] = contour[np.argmax(diff)]

	return result_contours


def gen_mats():
	marker_mat = []
	for col in range(80):
		for row in range(80):
			marker_mat.append([col,row,1])


	lena_mat = []
	for col in range(512):
		for row in range(512):
			lena_mat.append([col,row,1])

	return np.transpose(np.array(lena_mat)), np.transpose(np.array(marker_mat))
	
def correct_holes(img,msk):

	pass



def tag_image(img,contours,marker_matx):
	
	tags = []
	contours_tags = []
	for contour in contours:
		tag_img = pull_tag(img,contour,marker_matx)

		# cv2.imshow('frame',tag_img)
		# cv2.waitKey(-1)
		if tag_img is not None:
			contours_tags.append(contour)
			tags.append(decode(tag_img))
		
	added_frame = add_tag_imgs(img, contours_tags, tags)
	return added_frame


def pull_tag(img,contour_tag,marker_matx):
	draw = np.zeros([80,80,3],np.uint8)
	x = np.array([1, 100, 100, 1])
	y = np.array([1, 1, 100, 100])
	xp = np.array([contour_tag[0,0], contour_tag[1,0], contour_tag[2,0], contour_tag[3,0]])
	yp = np.array([contour_tag[0,1], contour_tag[1,1], contour_tag[2,1], contour_tag[3,1]])
	H = find_homography(x,y,xp,yp)

	xy_new = np.dot(H,marker_matx)
	print(H)
	print(xy_new.shape)
	xy_new = ((xy_new[0:2,:]/xy_new[2,:])).astype(int)
	print(xy_new)
	draw[marker_matx[0,:],marker_matx[1,:]] = img[xy_new[0,:],xy_new[1,:]]
	
	# for col in range(100):
	# 	for row in range(100):
	# 		xy_new = np.dot(H,np.array([col,row,1]))
	# 		xy_new = np.squeeze(xy_new[0,0:2]/xy_new[0,2])
	# 		if ((xy_new > [1920, 1080]).any() or (xy_new < 0).any()):
	# 			return None
	# 		draw[col,row] = img[int(xy_new[0,1]),int(xy_new[0,0])]
	return draw


def decode(marker_image):
	marker_img = cv2.resize(marker_image, (80, 80), interpolation = cv2.INTER_AREA)
	marker_img = cv2.cvtColor(marker_img,cv2.COLOR_BGR2GRAY)
	# marker_img = cv2.GaussianBlur(marker_img,(3,3),0)
	ret,th1 = cv2.threshold(marker_img,128,255,cv2.THRESH_BINARY)
	
	# kernel = np.ones((3,3),np.uint8)
	# th1 = cv2.erode(th1,kernel,iterations = 1)

	

	orientation_tags = np.array([np.sum(th1[21:26,21:26]),
						np.sum(th1[21:26,54:59]),
						np.sum(th1[54:59,21:26]),
						np.sum(th1[54:59,54:59])])
	# print(orientation_tags)

	data_tags =np.array([np.sum(th1[34:37,34:37])/9.0,
						np.sum(th1[34:37,44:47])/9.0,
						np.sum(th1[44:47,34:37])/9.0,
						np.sum(th1[44:47,44:47])/9.0])
	# print(data_tags)
	data_tags = (data_tags > 225).astype(int)

	corner = np.argmax(orientation_tags)
	if corner == 3:
		tag_ID = np.array([data_tags[2],data_tags[3],data_tags[1],data_tags[0]])
		tag_ID = data_tags
	elif corner == 2:
		tag_ID = np.array([data_tags[0],data_tags[2],data_tags[3],data_tags[1]])
		th1 = np.rot90(th1)
	elif corner == 0:
		tag_ID = np.array([data_tags[1],data_tags[0],data_tags[2],data_tags[3]])
		th1 = np.rot90(th1)
		th1 = np.rot90(th1)
	elif corner == 1:
		tag_ID = np.array([data_tags[3],data_tags[1],data_tags[0],data_tags[2]])
		th1 = np.rot90(th1)
		th1 = np.rot90(th1)
		th1 = np.rot90(th1)
	

	# th1[21:26,21:26] = 128
	# th1[21:26,54:59] = 128
	# th1[54:59,21:26] = 128
	# th1[54:59,54:59] = 128

	# th1[34:37,34:37] = 128
	# th1[34:37,44:47] = 128
	# th1[44:47,34:37] = 128
	# th1[44:47,44:47] = 128

	cv2.imshow('marker',th1)
	cv2.waitKey(1)


	# print(tag_ID)
	return tag_ID


def add_tag_imgs(img, contours, tags):

	font = cv2.FONT_HERSHEY_SIMPLEX
	fontScale = 1
	color = (255, 0, 0) 
	thickness = 2
	
	indx = 0
	for contour in contours:
		pnt = (max(contour[:,0]),min(contour[:,1]))
		image = cv2.putText(img, str(tags[indx]), pnt, font, fontScale, color, thickness, cv2.LINE_AA) 
		indx += 1
	return img


def superImpose(H,src,dest):

	# blank_image = np.zeros(dest.shape, np.uint8)
	# image_mask = np.zeros((dest.shape[0],dest.shape[1],1), np.uint8)

	
	# for col in range(src.shape[0]):
	# 	for row in range(src.shape[1]):
	# for col in range(50):
	# 	for row in range(50):
	# 		xy_new = np.dot(H,np.array([col,row,1]))
	# 		xy_new = xy_new[0,0:2]/xy_new[0,2]
			
	# 		dest[int(xy_new[0,1]),int(xy_new[0,0])] = src[col,row]

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