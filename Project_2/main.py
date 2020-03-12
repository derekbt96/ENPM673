import numpy as np 
import cv2
from matplotlib import pyplot as plt

'''
cd Documents/Documents/Aerospace/ENPM673/ENPM673/Project_2
'''


K_images =  np.array([[  1.15422732e+03,   0.00000000e+00,   6.71627794e+02],
				[  0.00000000e+00,   1.14818221e+03,   3.86046312e+02],
				[  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
dist_images = np.array([ -2.42565104e-01,  -4.77893070e-02,  -1.31388084e-03,  -8.79107779e-05,    2.20573263e-02])


newcameramtx_2images, roi = cv2.getOptimalNewCameraMatrix(K_images,dist_images,(1392,512),1, (1392,512))



part = 3

def main():
	

	if part == 1:
		capture = cv2.VideoCapture('Night Drive - 2689.mp4')
	elif part == 2:
		capture = cv2.VideoCapture('Night Drive - 2689.mp4')
		land
	elif part == 3:
		capture = cv2.VideoCapture('data_2/challenge_video.mp4')
		detector = Lane_detector_video()
	
	while(capture.isOpened()):

	
		ret, frame = capture.read()
		if frame is None:
			break
		
		# 1
		# result = improve_img(frame)
		# result2 = cv2.medianBlur(result, 5)


		# 2
		
		result = detector.spin(frame)
		result2 = detector.road
		
		
		# result = part2(frame)

		# result = cv2.resize(result, (800, 450), interpolation = cv2.INTER_AREA)
		# frame = cv2.resize(frame, (800, 450), interpolation = cv2.INTER_AREA)
		# cv2.imshow('frame',frame)
		cv2.imshow('result',result)
		cv2.imshow('result2',result2)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	capture.release()
	cv2.destroyAllWindows()



def part2(img):
	# h, w = img.shape[:2]
	# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1, (w,h))

	# Undistorting
	# dst = cv2.undistort(img, K, dist, None, newcameramtx)
	# dst = dst[250:250+h, 0:0+w]



	# blur to remove noise 
	# blurred = cv2.GaussianBlur(dst,(3,3),0)

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

	gray = clahe.apply(gray)
	gray = cv2.GaussianBlur(gray,(3,3),0)

	edges = cv2.Canny(gray,50,150,apertureSize = 3)

	return edges


	# HOUGH LINES
	# lines = cv2.HoughLines(edges,1,np.pi/180,200)
	# for x in range(0, len(lines)):    
	#   for rho, theta in lines[x]:
	#       a = np.cos(theta)
	#       b = np.sin(theta)
	#       x0 = a*rho
	#       y0 = b*rho
	#       x1 = int(x0 + 1000*(-b))
	#       y1 = int(y0 + 1000*(a))
	#       x2 = int(x0 - 1000*(-b))
	#       y2 = int(y0 - 1000*(a))

	#       cv2.line(dst,(x1,y1),(x2,y2),(0,0,255),2)

	# minLineLength = 25
	# maxLineGap = 10
	# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)

	# for x in range(0, len(lines)):    
	#     for x1,y1,x2,y2 in lines[x]:
	#         cv2.line(dst,(x1,y1),(x2,y2),(0,255,0),2)


	# LINE SEGMIENT DETECTOR
	# lsd = cv2.createLineSegmentDetector(0,0.8,0.6,2.0,22.5,0,0.7,1024 )

	# #Detect lines in the image
	# lines = lsd.detect(gray)[0] #Position 0 of the returned tuple are the detected lines

	# # #Draw detected lines in the image
	# dst = lsd.drawSegments(dst,lines)


	# ret,thresh_binary = cv2.threshold(gray,210,255,cv2.THRESH_BINARY)
	# erosion = cv2.erode(thresh_binary,(3,3),iterations = 1)
	# dilation = cv2.dilate(erosion,(3,3),iterations = 1)
	# open_ = cv2.morphologyEx(thresh_binary, cv2.MORPH_OPEN, (3,3))
	# open_ = cv2.dilate(open_,(3,3),iterations = 1)
	# mask = cv2.bitwise_and(gray, gray, mask = np.uint8(open_))
	# lines = lsd.detect(mask)[0] #Position 0 of the returned tuple are the detected lines

	# # #Draw detected lines in the image
	# mask = lsd.drawSegments(mask,lines)
	# return mask

class Lane_detector_video:
	def __init__(self):
		self.K =  np.array([[  1.15422732e+03,   0.00000000e+00,   6.71627794e+02],
					[  0.00000000e+00,   1.14818221e+03,   3.86046312e+02],
					[  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
		self.dist = np.array([ -2.42565104e-01,  -4.77893070e-02,  -1.31388084e-03,  -8.79107779e-05, 2.20573263e-02])

		self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(self.K,self.dist,(1280,720),1, (1280,720))
		self.turn = 0.0
		self.turn_R = 0.0
		self.turn_L = 0.0

		self.road = np.zeros((180,400),np.uint8)
		self.lane_mask = np.zeros((180,400),np.uint8)
		
		self.image = np.zeros((1280,720),np.uint8)
		
		lane_height = [470,680]
		lane_center = [670,640]
		lane_width = [140,500]
		self.lane_pnts = np.array([[lane_center[0]-lane_width[0],lane_height[0]],[lane_center[1]-lane_width[1],lane_height[1]],[lane_center[1]+lane_width[1],lane_height[1]],[lane_center[0]+lane_width[0],lane_height[0]]])
		self.current_lane = np.array([[700,470],[630,470],[300,690],[1050,690]])

		print(self.current_lane.shape)
		self.h = 400
		self.w = 180
		self.dst_pts = np.array([[1,1],[1,self.h],[self.w,self.h],[self.w, 1]])
		
		self.M, self.M_mask = cv2.findHomography(self.lane_pnts, self.dst_pts, cv2.RANSAC,5.0)
		self.Minv, self.Minv_mask = cv2.findHomography(self.dst_pts,self.lane_pnts, cv2.RANSAC,5.0)

		

	def spin(self,img):
		

		self.get_lane_points(img.copy())


		print(round(self.turn,2),' ',round(self.turn_R,2),' ',round(self.turn_L,2))

		
		return self.color_lane(img)
		

	def get_lane_points(self,img):

		# h, w = img.shape[:2]
		# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1, (w,h))

		# Undistorting
		img = cv2.undistort(img, self.K, self.dist, None, self.newcameramtx)
		

		road_og = cv2.warpPerspective(img, self.M, (self.w, self.h))
		road = road_og.copy()


		road_HSV = cv2.cvtColor(road, cv2.COLOR_BGR2HSV)
		road_LAB = cv2.cvtColor(road, cv2.COLOR_BGR2LAB)
		road_YCr = cv2.cvtColor(road, cv2.COLOR_BGR2YCrCb)

		
		(H, S, V) = cv2.split(road_HSV)
		(L, A, B) = cv2.split(road_LAB)
		# (Y, Cr, Cb) = cv2.split(road_YCr)
		(B, G, R) = cv2.split(road)
		# road = V

		# road = np.vstack([np.hstack([B, G, R, H, S, V]),np.hstack([L, A, B, Y, Cr, Cb])])

		# road = np.hstack([L, V, B, G, R])
		road = cv2.merge([L, V, R])

		# road = np.hstack([L, V, R])

		# lower_color_bounds = (50, 50, 50)
		# upper_color_bounds = (225,255,255)
		# mask = cv2.inRange(road,lower_color_bounds,upper_color_bounds)
		# mask_rgb = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
		# road = road & mask_rgb


		road = cv2.Canny(road, 50, 150)
		road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((1,3),np.uint8))
		road = cv2.morphologyEx(road, cv2.MORPH_OPEN, np.ones((5,1),np.uint8))
		road = cv2.erode(road,np.ones((3,3),np.uint8))

		self.lane_mask = road

		# road = cv2.GaussianBlur(road,(3,3),0)
		# road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8))

		left_lane_pnt = (40,400)
		right_lane_pnt = (144,400)

		center_coordinates = (120, 50)
		# cv2.circle(road_og, left_lane_pnt, 5, (0, 0, 255) , 2)
		# cv2.circle(road_og, right_lane_pnt, 5, (255, 0, 0) , 2)

		lines = cv2.HoughLines(road,1,np.pi/180,40)

		if lines is not None:
			lines = np.squeeze(lines,axis = 1)

			rotated_X = lines[:,0]*np.cos(lines[:,1])
			
			dist_thres = 40
			left_lines = lines[(np.abs(rotated_X - left_lane_pnt[0]) < dist_thres),:]
			right_lines = lines[(np.abs(rotated_X - right_lane_pnt[0]) < dist_thres),:]

			if left_lines.shape[0] > 0:
				for indx in range(left_lines.shape[0]):
					theta = left_lines[indx,1]
					rho = left_lines[indx,0]
					a = np.cos(theta)
					b = np.sin(theta)
					x0 = a*rho
					y0 = b*rho
					x1 = int(x0 + 1000*(-b))
					y1 = int(y0 + 1000*(a))
					x2 = int(x0 - 1000*(-b))
					y2 = int(y0 - 1000*(a))
					cv2.line(road_og,(x1,y1),(x2,y2),(0,0,255),1)
				
				left_lines[left_lines > .5*np.pi] = left_lines[left_lines > .5*np.pi]-np.pi
				self.turn_L = self.turn_L*.75 + .25*np.average(left_lines[:,1])

				x_L = left_lane_pnt[0] + int(self.h * np.sin(self.turn_L))
				cv2.circle(road_og, (x_L,0), 5, (0, 0, 255) , 2)

				xnew = np.dot(self.Minv,np.array([x_L,0,1]))
				self.current_lane[1,0] = int(xnew[0])


			if right_lines.shape[0] > 0:
				for indx in range(right_lines.shape[0]):
					theta = right_lines[indx,1]
					rho = right_lines[indx,0]
					a = np.cos(theta)
					b = np.sin(theta)
					x0 = a*rho
					y0 = b*rho
					x1 = int(x0 + 1000*(-b))
					y1 = int(y0 + 1000*(a))
					x2 = int(x0 - 1000*(-b))
					y2 = int(y0 - 1000*(a))
					cv2.line(road_og,(x1,y1),(x2,y2),(255,0,0),1)
			
				right_lines[right_lines > .5*np.pi] = right_lines[right_lines > .5*np.pi]-np.pi
				self.turn_R = self.turn_R*.5 + .5*np.average(right_lines[:,1])

				x_R = right_lane_pnt[0] + int(self.h * np.sin(self.turn_R))
				cv2.circle(road_og, (x_R,0), 5, (255, 0, 0) , 2)

				xnew = np.dot(self.Minv,np.array([x_R,0,1]))
				self.current_lane[0,0] = int(xnew[0])

			self.turn = self.turn*.9 + .1*(.5*self.turn_R + .5*self.turn_L)*180.0/np.pi



		self.road = road_og

	def color_lane(self,img):
		
		

		img2 = cv2.undistort(img, self.K, self.dist, None, self.newcameramtx)
		
		warped_road = cv2.warpPerspective(self.lane_mask, self.Minv, (1280, 720))
		
		img2[warped_road != 0] = (0,255,0)
		# img2[warped_road != (0,0,0)] = warped_road[warped_road != (0,0,0)]

		alpha = .4

		filled = cv2.fillPoly(img2.copy(), pts =[self.current_lane], color=(0,0,255))
		
		cv2.addWeighted(filled, alpha, img2, 1 - alpha,0, img2)
		# cv2.addWeighted(img, alpha, img2, 1 - alpha,0, img)
		
		font = cv2.FONT_HERSHEY_SIMPLEX 
		org = (680, 300)   
		image = cv2.putText(img2, str(round(self.turn,2)), org, font,  1, (40,40,40), 2, cv2.LINE_AA) 


		return img2



def part1(frame):
	gamma = 2
	corrected = gamma_correct(original, gamma)
	corrected = cv2.GaussianBlur(corrected,(5,5),0)

	gamma_inv = 1.0 / gamma
	lookup_table = np.array([((it / 255.0) ** gamma_inv) * 255

		for it in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table

	return cv2.LUT(image, lookup_table)


main()
