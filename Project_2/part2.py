import numpy as np 
import cv2
import sys
import os
PY3 = sys.version_info[0] == 3
dirpath = os.getcwd()
if PY3:
    xrange = range
from matplotlib import pyplot as plt
# from skimage import feature, color, transform, io

def main():
    # camera matrix
    K =  np.array([[  1.15422732e+03,   0.00000000e+00,   6.71627794e+02],
                    [  0.00000000e+00,   1.14818221e+03,   3.86046312e+02],
                    [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

    # distortion coefficients
    dist = np.array([ -2.42565104e-01,  -4.77893070e-02,  -1.31388084e-03,  
        -8.79107779e-05,    2.20573263e-02])

    for subdir, dirs, files in os.walk(dirpath + '/data'):
        files.sort()
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg") or filepath.endswith(".pgm") or filepath.endswith(".png") or filepath.endswith(".ppm"):

                # print(file)
                img = cv2.imread(filepath)

    # img = cv2.imread('test.png')
                h, w = img.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1, (w,h))

                # Undistorting
                dst = cv2.undistort(img, K, dist, None, newcameramtx)
                dst = dst[250:250+h, 0:0+w]

                

                # blur to remove noise 
                # blurred = cv2.GaussianBlur(dst,(3,3),0)

                gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

                gray = clahe.apply(gray)
                gray = cv2.GaussianBlur(gray,(3,3),0)

                edges = cv2.Canny(gray,50,150,apertureSize = 3)

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

                lsd = cv2.createLineSegmentDetector(0,0.8,0.6,2.0,22.5,0,0.7,1024 )

                #Detect lines in the image
                lines = lsd.detect(gray)[0] #Position 0 of the returned tuple are the detected lines

                # #Draw detected lines in the image
                dst = lsd.drawSegments(dst,lines)


                ret,thresh_binary = cv2.threshold(gray,210,255,cv2.THRESH_BINARY)
                erosion = cv2.erode(thresh_binary,(3,3),iterations = 1)
                dilation = cv2.dilate(erosion,(3,3),iterations = 1)
                open_ = cv2.morphologyEx(thresh_binary, cv2.MORPH_OPEN, (3,3))
                open_ = cv2.dilate(open_,(3,3),iterations = 1)
                mask = cv2.bitwise_and(gray, gray, mask = np.uint8(open_))
                lines = lsd.detect(mask)[0] #Position 0 of the returned tuple are the detected lines

                # #Draw detected lines in the image
                mask = lsd.drawSegments(mask,lines)

                cv2.imshow('image', mask)
                cv2.waitKey(0)
                # break
    

    # cap = cv2.VideoCapture('challenge_video.mp4')

    # while(cap.isOpened()):
    #   ret, img = cap.read()

    #   h, w = img.shape[:2]
    #   newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))

    #   # Undistorting
    #   dst = cv2.undistort(img, K, dist, None, newcameramtx)

    #   # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
 #  #       cv2.resizeWindow('image', 600, 600)
 #        cv2.imshow('image', dst)
 #        cv2.waitKey(2000)

            
        
main()

