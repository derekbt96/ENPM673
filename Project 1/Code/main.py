import numpy as np 
import cv2




def main():
	lena = cv2.imread('Lena.png')
	marker = cv2.imread('ref_marker.png')
	marker_grid = cv2.imread('ref_marker_grid.png')



	
	K = np.transpose(np.array([[1406.08415449821,0,0],
	[2.20679787308599, 1417.99930662800,0],
	[1014.13643417416, 566.347754321696,1]]))
	
	lena = cv2.imread('Lena.png')


	lena_mat, marker_mat = gen_mats()
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		

	tag_capture = cv2.VideoCapture('Tag0.mp4')
	# tag_capture = cv2.VideoCapture('Tag1.mp4')
	# tag_capture = cv2.VideoCapture('Tag2.mp4')
	# tag_capture = cv2.VideoCapture('multipleTags.mp4')
	# out = cv2.VideoWriter('multipleTags_tagged.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1920,1080))
	# out = cv2.VideoWriter('Tag0_tagged.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1920,1080))

	for a in range(100):
			ret, frame = tag_capture.read()

	while(tag_capture.isOpened()):

		
		# for i in range(10):
		# 	ret, frame = tag_capture.read()
	
		ret, frame = tag_capture.read()

		if frame is None:
			break
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = clahe.apply(gray)
	

		ret,th1 = cv2.threshold(gray,190,255,cv2.THRESH_BINARY)
		kernel = np.ones((3,3),np.uint8)
		th1 = cv2.erode(th1,kernel,iterations = 1)
		# edge = cv2.Canny(th1, 75, 200)


		contours, hierarchy = cv2.findContours(th1.copy() ,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
		# contours, hierarchy = cv2.findContours(edge.copy() ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# frame = cv2.drawContours(frame,contours,-1,(0,0,255),2)
		contours = refine_contours(contours,hierarchy)
		
		
		if len(contours) > 0:
			result, new_contours = tag_image(frame,contours,marker_mat.copy())
			if len(new_contours) > 0:
				result = superImpose(result,new_contours,lena,lena_mat.copy())
				# result = cv2.drawContours(result,draw_contours,-1,(0,0,255),2)
				pass
		else:
			result = frame
		
		# out.write(result)
		# cv2.imwrite('multipleTags_frame.png', result)

		result_R = cv2.resize(result, (800, 450), interpolation = cv2.INTER_AREA)
		# cv2.imshow('frame',result_R)
		# if cv2.waitKey(1) & 0xFF == ord('q'):
			# break
		break

	# out.release()
	tag_capture.release()
	cv2.destroyAllWindows()

	
	
	
	
def refine_contours(contours,hierarchy):

	area_bounds = [1000, 32000]
	result_contours = []
	contour_sum = np.array([])
	contour_areas = []
	sum_threshold = 100

	# print(hierarchy.shape)
	for contour_indx in range(len(contours)):
		area = cv2.contourArea(contours[contour_indx])
		

		arc_L = cv2.arcLength(contours[contour_indx], True)
		approxps = np.squeeze(cv2.approxPolyDP(contours[contour_indx], arc_L/30.0, True))
		approx = np.squeeze(approxps)
		if contour_sum.shape[0] > 0:
			closeness = (np.abs(contour_sum-np.sum(approx,axis=0)) < sum_threshold).any()
		else:
			closeness = False

		if area > area_bounds[0] and area < area_bounds[1] and approx.shape[0] == 4 and not closeness and hierarchy[0,contour_indx,-1] != -1:
			np.append(contour_sum,np.sum(approx,axis=0))
			contour_areas.append(area)


			# rect = np.zeros((4,2))
			TL = np.argmin(np.hypot(approx[:,0],approx[:,1]))
			TR = np.argmin(np.hypot(1920-approx[:,0],approx[:,1]))
			# BR = np.argmin(np.hypot(1920-approx[:,0],1080-approx[:,1]))
			# BL = np.argmin(np.hypot(approx[:,0],1080-approx[:,1]))
			# print(np.hypot(1920-approx[:,0],approx[:,1]))
			# print([TL,TR,BR,BL])
			# rect[0,:] = approx[TL,:]
			# rect[1,:] = approx[TR,:]
			# rect[2,:] = approx[BR,:]
			# rect[3,:] = approx[BL,:]
			# rect = rect.astype(int)
			# print(approx.shape)
			# print(rect.shape)
			# result_contours.append(rect)
			# print(rect)

			# result_contours.append(approx)
			# dxy = [approx[1,0] - approx[0,0], approx[0,1] - approx[1,1]]
			# if dxy[np.argmax(np.absolute(dxy))] > 0:
			if TL < TR and TL != 0:
				result_contours.append(approx)
			else:
				if TL == 0 and TR == 3:
					result_contours.append(np.flip(approx, 0))
				else:
					result_contours.append(approx)

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
	

def tag_image(img,contours,marker_matx):
	
	tags = []
	contours_tags = []
	final_contours = []
	indx_tag = 0
	for contour in contours:
	
		tag_img = pull_tag(img,contour,marker_matx)

		if tag_img is not None:
			contours_tags.append(contour)
			tag_temp, roll = decode(tag_img,indx_tag)

			tags.append(tag_temp)

			final_contours.append(np.roll(contour,-roll,axis=0))
			indx_tag += 1
	

	if len(tags) == 0:
		return img,None
	else:
		added_frame = add_tag_imgs(img, contours_tags, tags)
		return added_frame, final_contours


def pull_tag(img,contour_tag,marker_matx):

	draw = np.zeros([80,80,3],np.uint8)
	x = np.array([1, 80, 80, 1])
	y = np.array([1, 1, 80, 80])
	xp = np.array([contour_tag[0,0], contour_tag[1,0], contour_tag[2,0], contour_tag[3,0]])
	yp = np.array([contour_tag[0,1], contour_tag[1,1], contour_tag[2,1], contour_tag[3,1]])
	H = find_homography(x,y,xp,yp)

	xy_new = np.dot(H,marker_matx)
	xy_new = ((xy_new[0:2,:]/xy_new[2,:])).astype(int)

	if (xy_new < 0).any() or (xy_new[0,:] > 1920).any() or (xy_new[1,:] > 1080).any():
		return None
	draw[marker_matx[1,:],marker_matx[0,:]] = img[xy_new[1,:],xy_new[0,:]]
	
	return draw


def decode(marker_image,num):
	marker_img = cv2.resize(marker_image, (80, 80), interpolation = cv2.INTER_AREA)
	marker_img = cv2.cvtColor(marker_img,cv2.COLOR_BGR2GRAY)
	# marker_img = cv2.GaussianBlur(marker_img,(3,3),0)
	ret,th_m = cv2.threshold(marker_img,128,255,cv2.THRESH_BINARY)
	
	# kernel = np.ones((3,3),np.uint8)
	# th_m = cv2.erode(th_m,kernel,iterations = 1)

	

	orientation_tags = np.array([np.sum(th_m[21:26,21:26]),
						np.sum(th_m[21:26,54:59]),
						np.sum(th_m[54:59,21:26]),
						np.sum(th_m[54:59,54:59])])
	# print(orientation_tags)

	# print(data_tags)
	
	corner = np.argmax(orientation_tags)
	if corner == 2:
		th_m = np.rot90(th_m)
		roll = 1
	elif corner == 0:
		th_m = np.rot90(th_m)
		th_m = np.rot90(th_m)
		roll = 2
	elif corner == 1:
		th_m = np.rot90(th_m)
		th_m = np.rot90(th_m)
		th_m = np.rot90(th_m)
		roll = 3
	else:
		roll = 0
	
	# data_tags = np.zeros((4,1))
	data_tags = np.array([np.sum(th_m[43:48,33:38])/25.0,
						np.sum(th_m[43:48,43:48])/25.0,
						np.sum(th_m[33:38,43:48])/25.0,
						np.sum(th_m[33:38,33:38])/25.0])
	tag_ID = (data_tags > 254).astype(int)
	
	# th_m[0:5,35:45] = 128
	# th_m[21:26,21:26] = 128
	# th_m[21:26,54:59] = 128
	# th_m[54:59,21:26] = 128
	# th_m[54:59,54:59] = 128

	# if tag_ID[0]:
	# 	th_m[43:48,33:38] = 50
	# if tag_ID[1]:
	# 	th_m[43:48,43:48] = 50
	# if tag_ID[2]:
	# 	th_m[33:38,43:48] = 50
	# if tag_ID[3]:
	# 	th_m[33:38,33:38] = 50

	# cv2.imshow('marker'+str(num)+'_1',marker_img)
	# cv2.imshow('marker'+str(num)+'_2',th_m)
	# cv2.waitKey(1)
	cv2.imwrite('single_tag.png', th_m)


	# print(tag_ID)
	return tag_ID, roll


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


def superImpose(dest,contours,Lena,lena_matx):

	for contour in contours:
		x = np.array([1, 512, 512, 1])
		y = np.array([1, 1, 512, 512])
		xp = np.array([contour[0,0], contour[1,0], contour[2,0], contour[3,0]])
		yp = np.array([contour[0,1], contour[1,1], contour[2,1], contour[3,1]])
		H = find_homography(x,y,xp,yp)

		lena_temp = lena_matx.copy()
		xy_new = np.dot(H,lena_temp)
		xy_new = (xy_new[0:2,:]/xy_new[2,:]).astype(int)

		if not ((xy_new < 0).any() or (xy_new[0,:] > 1920).any() or (xy_new[1,:] > 1080).any()):
			dest[xy_new[1,:],xy_new[0,:]] = Lena[lena_temp[1,:],lena_temp[0,:]]
			
		

	return dest


def find_homography(x,y,xp,yp):



	A = np.matrix([ [-x[0], -y[0], -1, 0, 0, 0, x[0]*xp[0], y[0]*xp[0], xp[0]],
					[0, 0, 0, -x[0], -y[0], -1, x[0]*yp[0], y[0]*yp[0], yp[0]],
					[-x[1], -y[1], -1, 0, 0, 0, x[1]*xp[1], y[1]*xp[1], xp[1]],
					[0, 0, 0, -x[1], -y[1], -1, x[1]*yp[1], y[1]*yp[1], yp[1]],
					[-x[2], -y[2], -1, 0, 0, 0, x[2]*xp[2], y[2]*xp[2], xp[2]],
					[0, 0, 0, -x[2], -y[2], -1, x[2]*yp[2], y[2]*yp[2], yp[2]],
					[-x[3], -y[3], -1, 0, 0, 0, x[3]*xp[3], y[3]*xp[3], xp[3]],
					[0, 0, 0, -x[3], -y[3], -1, x[3]*yp[3], y[3]*yp[3], yp[3]]])

	# evl_AtA, V = np.linalg.eig(np.dot(np.transpose(A),A))
	# u, s, V = np.linalg.svd(A, full_matrices=True)
	# H = np.reshape(V[:,-1],[3,3])
	

	U, S, Vh = np.linalg.svd(A)
	l = Vh[-1, :] / Vh[-1, -1]
	H = np.reshape(l, (3, 3))

	
	return H




main()