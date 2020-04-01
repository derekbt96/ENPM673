import numpy as np 
import cv2
from matplotlib import pyplot as plt
from scipy.io import loadmat





def main():
	
	capture = cv2.VideoCapture('detectbuoy.avi')
	capture_mask = cv2.VideoCapture('buoy_mask.avi')

	color_seg = gmm()
	mask_gen = color_mask()
	frame_indx = 0
	while(True):

		
		ret, frame = capture.read()
		ret, frame_mask = capture_mask.read()
		frame_indx += 1
		if frame is None:
			break


		# mask1 = cv2.inRange(frame_mask, (20,0,0), (255,0,0))
		# mask2 = cv2.inRange(frame_mask, (0,20,0), (0,255,0))


		# (mask1, mask2, mask3) = cv2.split(frame_mask)
		
		# kernel = np.ones((9,9),np.uint8)

		# mask1 = cv2.inRange(frame_mask, (0,0,225), (10,10,255))
		# mask1 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, kernel)

		# mask2 = cv2.inRange(frame_mask, (0,50,0), (10,255,10))
		# mask2 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, kernel)

		# mask3 = cv2.inRange(frame_mask, (50,0,0), (255,10,10))
		# mask3 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, kernel)

		# mask_full = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))

		mask1 = mask_gen.get_mask(frame,frame_indx,1)
		mask2 = mask_gen.get_mask(frame,frame_indx,2)
		mask3 = mask_gen.get_mask(frame,frame_indx,3)
		
		result = cv2.merge([mask1,mask2,mask3])
		

		# result = cv2.bitwise_and(frame,frame, mask = mask1)
		
		
		
		# mask2 = mask2 > 200
		# mask3 = mask3 > 200
		# print('2')
		# frame[mask1] == (255,0,0)
		# frame[mask2] == (0,255,0)
		# frame[mask3] == (0,0,255)
		# print('3')
		
		# color_seg.spin(frame)

		# cv2.imshow('mask1',mask1)
		# cv2.imshow('mask2',mask2)
		# cv2.imshow('mask3',mask3)
		cv2.imshow('result',result)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
	capture.release()
	capture_mask.release()
	cv2.destroyAllWindows()



class color_mask:
	def __init__(self):
		temp = loadmat('vid_points.mat')
		self.pnts = temp['image_points']
		
		self.wnd = 25
		self.thres = [80,80,40]
		

	def get_mask(self,img,img_num,mask_num):
		x = self.pnts[img_num,((mask_num-1)*2)]
		y = self.pnts[img_num,((mask_num-1)*2)+1]
		
		color = img[x,y,:]

		temp_wnd = self.wnd
		if img_num > 135:
			temp_wnd = 40
		
		img_red = img[max(x-temp_wnd,0):min(x+temp_wnd,640),max(y-temp_wnd,0):min(y+temp_wnd,480),:]
		

		upper_b = (color[0]-self.thres[0],color[1]-self.thres[1],color[2]-self.thres[2])
		lower_b = (color[0]+self.thres[0],color[1]+self.thres[1],color[2]+self.thres[2])
		print(upper_b)
		print(lower_b)
		mask = cv2.inRange(img_red, lower_b, upper_b)
		
		mask_final = np.zeros((480,640),np.uint8)
		
		mask_final[max(x-wnd,0):min(x+wnd,640),max(y-wnd,0):min(y+wnd,480)] = mask
		
		return mask_final







class gmm:
	def __init__(self):
		self.K =  np.array([[  1.15422732e+03,   0.00000000e+00,   6.71627794e+02],
					[  0.00000000e+00,   1.14818221e+03,   3.86046312e+02],
					[  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
		self.dist = np.array([ -2.42565104e-01,  -4.77893070e-02,  -1.31388084e-03,  -8.79107779e-05, 2.20573263e-02])

		self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(self.K,self.dist,(1280,720),1, (1280,720))
		
		
		

	def spin(self,img):
		
		return img

		
		
	def change_colorspace(self,img):
		HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		YCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
		return HSV

main()



# cd Documents/Documents/Aerospace/ENPM673/ENPM673/Project_3