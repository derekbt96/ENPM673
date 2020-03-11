import numpy as np 
import cv2
from matplotlib import pyplot as plt

'''
cd Documents/Documents/Aerospace/ENPM673/ENPM673/Project_2
'''

def main():
	

	# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

	capture = cv2.VideoCapture('data_2/challenge_video.mp4')
	# capture = cv2.VideoCapture('Night Drive - 2689.mp4')
	
	# for a in range(20):
		# ret, frame = capture.read()

	while(capture.isOpened()):

	
		ret, frame = capture.read()
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		if frame is None:
			break
		
		# 1
		# result = improve_img(frame)
		# result2 = cv2.medianBlur(result, 5)

		# 2
		
		# lane_points = get_lane_points(frame)
		# lane_height = [425,700]
		# lane_center = [670,640]
		# lane_width = [110,600]
		# lane_points = np.array([[lane_center[0]-lane_width[0],475],[lane_center[1]-lane_width[1],700],[lane_center[1]+lane_width[1],700],[lane_center[0]+lane_width[0],475]])

		# result = tag_lane(frame,lane_points)

		result = get_lane_points(frame)
		

		# result = cv2.resize(result, (800, 450), interpolation = cv2.INTER_AREA)
		# frame = cv2.resize(frame, (800, 450), interpolation = cv2.INTER_AREA)
		cv2.imshow('frame',frame)
		cv2.imshow('result',result)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	capture.release()
	cv2.destroyAllWindows()



def get_lane_points(img):

	lane_height = [425,700]
	lane_center = [670,640]
	lane_width = [110,600]
	lane_pnts = np.array([[lane_center[0]-lane_width[0],475],[lane_center[1]-lane_width[1],700],[lane_center[1]+lane_width[1],700],[lane_center[0]+lane_width[0],475]])
	h = 300
	w = 150
	dst_pts = np.array([[1,1],[1,h],[w,h],[w, 1]])

	M, mask = cv2.findHomography(lane_pnts, dst_pts, cv2.RANSAC,5.0)
	
	road = cv2.warpPerspective(img, M, (w, h))

	# (blue, green, red) = cv2.split(road)
	# red = cv2.equalizeHist(red)
	# green = cv2.equalizeHist(green)
	# blue = cv2.equalizeHist(blue)
	# road = cv2.merge([blue, green, red])

	road_HSV = cv2.cvtColor(road, cv2.COLOR_BGR2HSV)
	road_LAB = cv2.cvtColor(road, cv2.COLOR_BGR2LAB)
	road_YCr = cv2.cvtColor(road, cv2.COLOR_BGR2YCrCb)

	# lower_color_bounds = (100, 0, 0)
	# upper_color_bounds = (225,80,80)

	(H, S, V) = cv2.split(road_HSV)
	(L, A, B) = cv2.split(road_LAB)
	(Y, Cr, Cb) = cv2.split(road_YCr)
	(B, G, R) = cv2.split(road)
	# road = V

	road = np.vstack([np.hstack([B, G, R, H, S, V]),np.hstack([L, A, B, Y, Cr, Cb])])
	
	# road = np.hstack([H, S, V])
	# gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	# mask = cv2.inRange(frame,lower_color_bounds,upper_color_bounds )
	# mask_rgb = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
	# frame = frame & mask_rgb

	# road = cv2.GaussianBlur(road,(3,3),0)
	road = cv2.Canny(road, 75, 200)
	road = cv2.morphologyEx(road, cv2.MORPH_OPEN, (5,5))

	# lines = cv2.HoughLines(road,1,np.pi/180,50)
	# for rho,theta in lines[0]:
	#     a = np.cos(theta)
	#     b = np.sin(theta)
	#     x0 = a*rho
	#     y0 = b*rho
	#     x1 = int(x0 + 1000*(-b))
	#     y1 = int(y0 + 1000*(a))
	#     x2 = int(x0 - 1000*(-b))
	#     y2 = int(y0 - 1000*(a))

	#     cv2.line(road,(x1,y1),(x2,y2),(0,0,125),2)


	return road
		

def improve_img(frame):
	(blue, green, red) = cv2.split(frame)


	
	# red_new = clahe.apply(red)
	# green_new = clahe.apply(green)
	# blue_new = clahe.apply(blue)
	# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(red)
	# print(min_val,' ',max_val)

	# red = cv2.equalizeHist(red)
	# green = cv2.equalizeHist(green)
	# blue = cv2.equalizeHist(blue)


	# red = red*5.0
	# green = green*5.0
	# blue = blue*5.0
	thres = 120
	
	# red[thres_mask] = 0
	# green[thres_mask] = 0
	# blue[thres_mask] = 0

	result = cv2.merge([blue, green, red])

	result = cv2.convertScaleAbs(result, alpha=5.0, beta=15.0)
	result = cv2.GaussianBlur(result,(3,3),0)

	
	# hist,bins = np.histogram(frame.flatten(),256,[0,256])
	# cdf = hist.cumsum()
	# cdf_normalized = cdf * float(hist.max()) / cdf.max()
	# plt.plot(cdf_normalized, color = 'b')
	# plt.hist(frame.flatten(),256,[0,256], color = 'r')
	# plt.xlim([0,256])
	# plt.legend(('cdf','histogram'), loc = 'upper left')
	# plt.show()

	# cdf_m = np.ma.masked_equal(cdf,0)
	# cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	# cdf = np.ma.filled(cdf_m,0).astype('uint8')

	# result = cdf[frame]
	
	# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result[:,:,1])
	# print(min_val,' ',max_val)

	# plt.hist(frame.ravel(),256,[0,256]); plt.show()
	# result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
	return result


def tag_lane(img,lane_pnts):
	# draw = np.zeros([80,80,3],np.uint8)
	h = 300
	w = 150
	dst_pts = np.array([[1,1],[1,h],[w,h],[w, 1]])

	M, mask = cv2.findHomography(lane_pnts, dst_pts, cv2.RANSAC,5.0)
	
	road = cv2.warpPerspective(img, M, (w, h))
	road = cv2.GaussianBlur(road,(3,3),0)

	# (blue, green, red) = cv2.split(road)
	# red = cv2.equalizeHist(red)
	# green = cv2.equalizeHist(green)
	# blue = cv2.equalizeHist(blue)
	# road = cv2.merge([blue, green, red])

	alpha = .5
	filled = cv2.fillPoly(img.copy(), pts =[lane_pnts], color=(0,0,255))
	cv2.addWeighted(filled, alpha, img, 1 - alpha,0, img)
	
	return road


main()
