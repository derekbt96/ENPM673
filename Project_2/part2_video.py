import numpy as np 
import cv2
from matplotlib import pyplot as plt

def main():
    
    capture = cv2.VideoCapture('challenge_video.mp4')

    while(capture.isOpened()):
    
        ret, frame = capture.read()

        if frame is None:
            break

        lane_points = np.array([[500,475],[50,700],[1280-50,700],[1280-500,475]])
        result = tag_lane(frame,lane_points)

        cv2.imshow('frame',frame)
        cv2.imshow('result',result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

def improve_img(frame):
    (blue, green, red) = cv2.split(frame)

    red_new = clahe.apply(red)
    green_new = clahe.apply(green)
    blue_new = clahe.apply(blue)

    thres = 120
    
    red[thres_mask] = 0
    green[thres_mask] = 0
    blue[thres_mask] = 0

    result = cv2.merge([blue, green, red])

    result = cv2.convertScaleAbs(result, alpha=5.0, beta=15.0)
    result = cv2.GaussianBlur(result,(3,3),0)

    return result

def tag_lane(img,lane_pnts):
    h = 300
    w = 150
    dst_pts = np.array([[1,1],[1,h],[w,h],[w, 1]])

    M, mask = cv2.findHomography(lane_pnts, dst_pts, cv2.RANSAC,5.0)
    
    road = cv2.warpPerspective(img, M, (w, h))

    hsv = cv2.cvtColor(road, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower = np.array([14,44,0])
    upper = np.array([51,151,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower, upper)

    # Bitwise-AND mask and original image
    yellow = cv2.bitwise_and(road,road, mask= mask)
    yellow = cv2.cvtColor(yellow, cv2.COLOR_BGR2GRAY)
    # yellow[0:h,80:150] = np.zeros((h,150-80))

    alpha = .5
    filled = cv2.fillPoly(img.copy(), pts =[lane_pnts], color=(0,0,255))
    cv2.addWeighted(filled, alpha, img, 1 - alpha,0, img)

    gray = cv2.cvtColor(road,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    gray = clahe.apply(gray)
    # gray = cv2.medianBlur(gray,(3,3),0)
    gray = cv2.bilateralFilter(gray,3,20,30)

    lsd = cv2.createLineSegmentDetector(0)

    ret,thresh_binary = cv2.threshold(gray,210,255,cv2.THRESH_BINARY)
    thresh_binary[0:h,0:110] = np.zeros((h,110))

    thresh_binary = cv2.add(thresh_binary,yellow)
    
    # erosion = cv2.erode(thresh_binary,(3,3),iterations = 1)
    # dilation = cv2.dilate(erosion,(3,3),iterations = 1)
    # open_ = cv2.morphologyEx(thresh_binary, cv2.MORPH_OPEN, (3,3))
    # open_ = cv2.dilate(open_,(3,3),iterations = 1)
    # mask = cv2.bitwise_and(gray, gray, mask = np.uint8(open_))
    lines = lsd.detect(thresh_binary)[0] #Position 0 of the returned tuple are the detected lines

    # for x in range(0, len(lines)):    
    #     for x1, y1, x2, y2 in lines[0]:
    #         m = (y2-y1)/(x2-x1)
    #         c = y2 - m*x2
    #         x_ = -c/m
    #         y_ = 0
    #         x_1 = (h-c)/m
    #         y_1 = h
    #         cv2.line(road,(int(x_),int(y_)),(int(x_1),int(y_1)),(0,0,255),2)
    
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
    #         cv2.line(road,(x1,y1),(x2,y2),(0,255,0),2)

    

    
    # #Draw detected lines in the image
    dst = lsd.drawSegments(road,lines)

    # print(lines)

    # return dst[0:h,110:150]

    return road

main()
