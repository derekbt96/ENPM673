import numpy as np 
import cv2
from matplotlib import pyplot as plt




def main():
	
	capture = cv2.VideoCapture('detectbuoy.avi')

	color_seg = gmm()

	# out = cv2.VideoWriter('road_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (800,800))
	while(True):

	
		ret, frame = capture.read()
		if frame is None:
			break

		# color_seg.spin(frame)
		

		result = cv2.resize(frame, (800, 400), interpolation = cv2.INTER_AREA)
		# out.write(result2)
		cv2.imshow('result',result)
		# cv2.imshow('result2',result2)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	out.release()
	capture.release()
	cv2.destroyAllWindows()




class gmm:
	def __init__(self):
		self.K =  np.array([[  1.15422732e+03,   0.00000000e+00,   6.71627794e+02],
					[  0.00000000e+00,   1.14818221e+03,   3.86046312e+02],
					[  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
		self.dist = np.array([ -2.42565104e-01,  -4.77893070e-02,  -1.31388084e-03,  -8.79107779e-05, 2.20573263e-02])

		self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(self.K,self.dist,(1280,720),1, (1280,720))
		

		

	def spin(self,img):
		
		return img
		

main()
