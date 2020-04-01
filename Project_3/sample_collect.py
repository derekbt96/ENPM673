# segments out the pixels of interest within the bounding box you choose; 
# Note: Pl choose the roi such that it mostly encompasses the buoys else you might get bad circles!
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D  
import sys
import os
PY3 = sys.version_info[0] == 3
dirpath = os.getcwd()

# Save/Load File name for RGB values
filename = 'roi/yellow.npy'
# load current training dataset # comment it out if you are initializing npy file
temp = np.load(filename, None, True, True, 'ASCII')
samples = []
# iterate over all images
for subdir, dirs, files in os.walk(dirpath + '/images'):
	files.sort()
	for file in files:
		filepath = subdir + os.sep + file
		if filepath.endswith(".jpg") or filepath.endswith(".pgm") or filepath.endswith(".png") or filepath.endswith(".ppm"):

			# Selected training img
			imgname = filepath
			# load image
			# print(imgname)
			img = cv2.imread(imgname)
			
			# Select rectangular regions of interest.  
			# Be careful not to select bad data or this will make our life hell
			# Stores all samples that have been selected
			# print('before')
			# print(len(samples))
			roi = cv2.selectROIs('Window',img)
			roi_len  = len(roi)
			if roi_len: 
				print('roi selected {}'.format(roi_len))
				for i in range(roi_len):
					samples.append(roi[i])
				
			print('samples collected thus far {}'.format(len(samples)))
			print(samples)
			cv2.destroyAllWindows()

## Initialize RGB values # do this just once to create an npy file, 
## or if you want to do data collection from the beginning
B = np.array([])
G = np.array([])
R = np.array([])
## comment out the below if you are reusing existing data in your npy file
# B = temp[0]
# G = temp[1]
# R = temp[2]
# print(samples)

for sample in samples:
	# sample has a starting point, top left corner of box, and a width, and a height
	yS = slice(sample[0],sample[0]+sample[2],1)
	xS = slice(sample[1],sample[1]+sample[3],1)
	# dump in the pixels corresponding to the slice
	sampleImg = img[xS,yS,:]
	h = sampleImg.shape[0]
	w = sampleImg.shape[1]
	
	image = sampleImg.copy()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	cv2.imshow("gray", gray)
# 	cv2.waitKey(1000)
	detected_circles = cv2.HoughCircles(gray,  
		cv2.HOUGH_GRADIENT, 1, 5, param1 = 50, 
		param2 = 15, minRadius = 10, maxRadius = 200) 
	# print('detected_circles')
	# print(detected_circles)
	
	# Draw circles that are detected. 
	if detected_circles is not None: 
	# print('circle!!')
		# Convert the circle parameters a, b and r to integers. 
		detected_circles = np.uint16(np.around(detected_circles)) 
		
		# for pt in detected_circles[0, :]: 
		a, b, r = detected_circles[0, 0, 0], detected_circles[0, 0, 1], detected_circles[0, 0, 2]             
		# Draw the circumference of the circle. 
		cv2.circle(sampleImg, (a, b), r, (0, 255, 0), 2) 
		cv2.imshow("sampleImg", sampleImg) 
		cv2.waitKey(1000) 

		windw = sampleImg.copy()
		h = sampleImg.shape[0]
		w = sampleImg.shape[1]
		for y in range (h):
			for x in range (w):
				if (np.sqrt(np.square(x-a) +np.square(y-a)) < r):
					B = np.append(B,windw[y,x,0])
					G = np.append(G,windw[y,x,1])
					R = np.append(R,windw[y,x,2])

# Save newly added training points to dataset
np.save(filename,np.vstack((B,G,R)))

# Make a 3D ScaTTER Of RGB values
fig = plt.figure()
ax = Axes3D(fig)

for i in range(0,R.shape[0]-1):
	# print('{} {} {}'.format (R[i], G[i], B[i]))
	ax.scatter(R[i], G[i], B[i],color=np.array([R[i], G[i], B[i]])/255.0)

ax.set_xlabel('B')
ax.set_ylabel('G')
ax.set_zlabel('R')

plt.show()
