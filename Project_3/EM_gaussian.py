import numpy as np 
import cv2
from matplotlib import pyplot as plt
from functions import color_mask, color_data




def main():
	
	num_gaussians = 3
	np.random.seed(1)
	rand_data = np.random.random((2,num_gaussians))
	means = rand_data[0,:]*10.0
	covs = rand_data[1,:]*3.0
	
	print(means)
	print(covs)

	dat = np.array([])
	for k in range(num_gaussians):
		dat = np.append(dat,np.random.randn(50,1)*covs[k]+means[k])


	fig = plt.figure()
	plt.hist(dat, bins=256, range=(-5, 15), fc='b', ec='b')
	plt.show()

	
main()
