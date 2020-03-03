import numpy as np 
import cv2
from matplotlib import pyplot as plt

'''
cd Documents/Documents/Aerospace/ENPM673/ENPM673/Project_2
'''

def main():
	lena = cv2.imread('Lena.png')
	marker = cv2.imread('ref_marker.png')
	marker_grid = cv2.imread('ref_marker_grid.png')


	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

	# capture = cv2.VideoCapture('data_2/challenge_video.mp4')
	capture = cv2.VideoCapture('Night Drive - 2689.mp4')
	
	for a in range(20):
		ret, frame = capture.read()

	while(capture.isOpened()):

	
		ret, frame = capture.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		if frame is None:
			break
		
		(blue, green, red) = cv2.split(frame)


		
		red_new = clahe.apply(red)
		green_new = clahe.apply(green)
		blue_new = clahe.apply(blue)
		# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(red)
		# print(min_val,' ',max_val)

		# red = cv2.equalizeHist(red)
		# green = cv2.equalizeHist(green)
		# blue = cv2.equalizeHist(blue)


		# red = red*5.0
		# green = green*5.0
		# blue = blue*5.0
		thres = 120
		
		red[thres_mask] = 0
		green[thres_mask] = 0
		blue[thres_mask] = 0

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


		result_R = cv2.resize(result, (800, 450), interpolation = cv2.INTER_AREA)
		frame_R = cv2.resize(frame, (800, 450), interpolation = cv2.INTER_AREA)
		cv2.imshow('result',result_R)
		cv2.imshow('frame',frame_R)
		if cv2.waitKey(-1) & 0xFF == ord('q'):
			break
		break

	capture.release()
	cv2.destroyAllWindows()


main()
