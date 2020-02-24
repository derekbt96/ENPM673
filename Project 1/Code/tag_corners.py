import numpy as np
import cv2

cap = cv2.VideoCapture('multipleTags.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    img_orig=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    img_orig = clahe.apply(img_orig)
    # binary threshold 
    ret,thresh_binary = cv2.threshold(img_orig,210,255,cv2.THRESH_BINARY)
    im, cnts, hierarchy = cv2.findContours(thresh_binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 

    # im, cnts, hierarchy = cv2.findContours(img_orig[0:1080,0:1920], cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt_area = []
    cnt_num = []
    for c in cnts:
        cnt_area.append(cv2.contourArea(c))

    cnt_num = np.argsort(cnt_area)
    cnt_area.sort()
    print(len(cnt_area))
    # large_cnts = np.zeros(np.shape(mask))
    fresh_im = np.zeros(np.shape(img_orig))

    if len(cnt_area) < 9:
        ## only draw the second largest contour 
        cv2.drawContours(fresh_im, cnts, cnt_num[len(cnt_num)-1-1], (255, 255, 255), -1)
        c = cnts[cnt_num[len(cnt_num)-1-1]]

        mask = cv2.bitwise_and(frame, frame, mask = np.uint8(fresh_im))
        
        epsilon = 0.1*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)
     
        if approx.shape[0] == 4: 
        	cv2.circle(mask, (approx[0,0,0], approx[0,0,1]), 8, (0, 255, 0), -1)
        	cv2.circle(mask, (approx[1,0,0], approx[1,0,1]), 8, (0, 255, 0), -1)
        	cv2.circle(mask, (approx[2,0,0], approx[2,0,1]), 8, (0, 255, 0), -1)
        	cv2.circle(mask, (approx[3,0,0], approx[3,0,1]), 8, (0, 255, 0), -1)

    else:

        cv2.drawContours(fresh_im, cnts, cnt_num[len(cnt_num)-1-3], (255, 255, 255), -1)
        cv2.drawContours(fresh_im, cnts, cnt_num[len(cnt_num)-1-4], (255, 255, 255), -1)
        cv2.drawContours(fresh_im, cnts, cnt_num[len(cnt_num)-1-5], (255, 255, 255), -1)

        c = cnts[cnt_num[len(cnt_num)-1-3]]
        c1 = cnts[cnt_num[len(cnt_num)-1-4]]
        c2 = cnts[cnt_num[len(cnt_num)-1-5]]

        # # mask of the largest contours 
        mask = cv2.bitwise_and(frame, frame, mask = np.uint8(fresh_im))
        
        epsilon = 0.1*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)

        epsilon1 = 0.1*cv2.arcLength(c1,True)
        approx1 = cv2.approxPolyDP(c1,epsilon1,True)

        epsilon2 = 0.1*cv2.arcLength(c2,True)
        approx2 = cv2.approxPolyDP(c2,epsilon2,True)
       
        if approx.shape[0] == 4 and approx1.shape[0] == 4 and approx2.shape[0] == 4: 
            cv2.circle(mask, (approx[0,0,0], approx[0,0,1]), 8, (0, 255, 0), -1)
            cv2.circle(mask, (approx[1,0,0], approx[1,0,1]), 8, (0, 255, 0), -1)
            cv2.circle(mask, (approx[2,0,0], approx[2,0,1]), 8, (0, 255, 0), -1)
            cv2.circle(mask, (approx[3,0,0], approx[3,0,1]), 8, (0, 255, 0), -1)
            cv2.circle(mask, (approx1[0,0,0], approx1[0,0,1]), 8, (0, 255, 0), -1)
            cv2.circle(mask, (approx1[1,0,0], approx1[1,0,1]), 8, (0, 255, 0), -1)
            cv2.circle(mask, (approx1[2,0,0], approx1[2,0,1]), 8, (0, 255, 0), -1)
            cv2.circle(mask, (approx1[3,0,0], approx1[3,0,1]), 8, (0, 255, 0), -1)
            cv2.circle(mask, (approx2[0,0,0], approx2[0,0,1]), 8, (0, 255, 0), -1)
            cv2.circle(mask, (approx2[1,0,0], approx2[1,0,1]), 8, (0, 255, 0), -1)
            cv2.circle(mask, (approx2[2,0,0], approx2[2,0,1]), 8, (0, 255, 0), -1)
            cv2.circle(mask, (approx2[3,0,0], approx2[3,0,1]), 8, (0, 255, 0), -1)
    
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600,600)
    cv2.imshow('image',mask)

    # cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image1', 600,600)
    # cv2.imshow('image1',frame[0:1080,0:1920])

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
