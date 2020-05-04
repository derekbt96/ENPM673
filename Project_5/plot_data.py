import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import os
from functions import RansacFundamental, EpipolarLines, getCameraPose, Linear,checkCheirality, camera_pose, recoverPose



pose = camera_pose()


pose.X_log = np.load('logs/straight_opencv.npy')

pose.plot()