import numpy as np
import cv2

def gen_mats():
    marker_mat = []
    for col in range(80):
        for row in range(80):
            marker_mat.append([col,row,1])


    lena_mat = []
    for col in range(512):
        for row in range(512):
            lena_mat.append([col,row,1])

    return np.transpose(np.array(lena_mat)), np.transpose(np.array(marker_mat))
    
def correct_holes(img,msk):

    pass

def tag_image(img,contours,marker_matx):
    
    tags = []
    contours_tags = []
    for contour in contours:
        tag_img = pull_tag(img,contour,marker_matx)

        # cv2.imshow('frame',tag_img)
        # cv2.waitKey(-1)
        if tag_img is not None:
            contours_tags.append(contour)
            tags.append(decode(tag_img))
    
    added_frame = add_tag_imgs(img, contours_tags, tags)
    return added_frame

def pull_tag(img,contour_tag,marker_matx):
    # print(contour_tag)
    # print(contour_tag[0,0,0])
    # print(contour_tag[1,0,1])
    draw = np.zeros([80,80,3],np.uint8)
    x = np.array([1, 80, 80, 1])
    y = np.array([1, 1, 80, 80])
    xp = np.array([contour_tag[0,0,0], contour_tag[1,0,0], contour_tag[2,0,0], contour_tag[3,0,0]])
    yp = np.array([contour_tag[0,0,1], contour_tag[1,0,1], contour_tag[2,0,1], contour_tag[3,0,1]])
    H = find_homography(x,y,xp,yp)
    xy_new = np.dot(H,marker_matx)
    # print(H)
    # print(marker_matx)
    
    xy_new = ((xy_new[0:2,:]/xy_new[2,:])).astype(int)
    if (xy_new < 0).any() or (xy_new[0,:] > 1920).any() or (xy_new[1,:] > 1080).any():
        return None
    draw[marker_matx[1,:],marker_matx[0,:]] = img[xy_new[1,:],xy_new[0,:]]
        
    # for col in range(100):
    #   for row in range(100):
    #       xy_new = np.dot(H,np.array([col,row,1]))
    #       xy_new = np.squeeze(xy_new[0,0:2]/xy_new[0,2])
    #       if ((xy_new > [1920, 1080]).any() or (xy_new < 0).any()):
    #           return None
    #       draw[col,row] = img[int(xy_new[0,1]),int(xy_new[0,0])]
    return draw


def decode(marker_image):
    marker_img = cv2.resize(marker_image, (80, 80), interpolation = cv2.INTER_AREA)
    marker_img = cv2.cvtColor(marker_img,cv2.COLOR_BGR2GRAY)
    # marker_img = cv2.GaussianBlur(marker_img,(3,3),0)
    ret,th1 = cv2.threshold(marker_img,128,255,cv2.THRESH_BINARY)
    
    # kernel = np.ones((3,3),np.uint8)
    # th1 = cv2.erode(th1,kernel,iterations = 1)

    

    orientation_tags = np.array([np.sum(th1[21:26,21:26]),
                        np.sum(th1[21:26,54:59]),
                        np.sum(th1[54:59,21:26]),
                        np.sum(th1[54:59,54:59])])
    # print(orientation_tags)

    data_tags =np.array([np.sum(th1[34:37,34:37])/9.0,
                        np.sum(th1[34:37,44:47])/9.0,
                        np.sum(th1[44:47,34:37])/9.0,
                        np.sum(th1[44:47,44:47])/9.0])
    # print(data_tags)
    data_tags = (data_tags > 225).astype(int)

    corner = np.argmax(orientation_tags)
    if corner == 3:
        tag_ID = np.array([data_tags[2],data_tags[3],data_tags[1],data_tags[0]])
        tag_ID = data_tags
    elif corner == 2:
        tag_ID = np.array([data_tags[0],data_tags[2],data_tags[3],data_tags[1]])
        th1 = np.rot90(th1)
    elif corner == 0:
        tag_ID = np.array([data_tags[1],data_tags[0],data_tags[2],data_tags[3]])
        th1 = np.rot90(th1)
        th1 = np.rot90(th1)
    elif corner == 1:
        tag_ID = np.array([data_tags[3],data_tags[1],data_tags[0],data_tags[2]])
        th1 = np.rot90(th1)
        th1 = np.rot90(th1)
        th1 = np.rot90(th1)

    cv2.imshow('marker',th1)
    cv2.waitKey(1)

    return tag_ID


def add_tag_imgs(img, contours, tags):

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0) 
    thickness = 2
    
    indx = 0
    for contour in contours:
        pnt = (max(contour[:,:,0]),min(contour[:,:,1]))
        image = cv2.putText(img, str(tags[indx]), pnt, font, fontScale, color, thickness, cv2.LINE_AA) 
        indx += 1
    return img


def superImpose(H,src,dest):

    # blank_image = np.zeros(dest.shape, np.uint8)
    # image_mask = np.zeros((dest.shape[0],dest.shape[1],1), np.uint8)

    
    # for col in range(src.shape[0]):
    #   for row in range(src.shape[1]):
    # for col in range(50):
    #   for row in range(50):
    #       xy_new = np.dot(H,np.array([col,row,1]))
    #       xy_new = xy_new[0,0:2]/xy_new[0,2]
            
    #       dest[int(xy_new[0,1]),int(xy_new[0,0])] = src[col,row]

    # for col in range(src.shape[0]):
        # for row in range(src.shape[1]):
    for col in range(50):
        for row in range(50):
            xy_new = np.dot(H,np.array([col,row,.5]))
            xy_new = xy_new[0,0:2]/xy_new[0,2]
            
            dest[int(xy_new[0,1]),int(xy_new[0,0])] = src[col,row]
    
    # warp = cv2.warpPerspective(src, H, (dest.shape[1],dest.shape[0]))
    # indx = np.where(warp != [0,0,0])
    # dest[indx] = warp[indx]


    destR = cv2.resize(dest, (800, 450), interpolation = cv2.INTER_AREA)
    while True:
        cv2.imshow('frame',dest)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return dest


def find_homography(x,y,xp,yp):
    A = np.matrix([ [-x[0], -y[0], -1, 0, 0, 0, x[0]*xp[0], y[0]*xp[0], xp[0]],
                    [0, 0, 0, -x[0], -y[0], -1, x[0]*yp[0], y[0]*yp[0], yp[0]],
                    [-x[1], -y[1], -1, 0, 0, 0, x[1]*xp[1], y[1]*xp[1], xp[1]],
                    [0, 0, 0, -x[1], -y[1], -1, x[1]*yp[1], y[1]*yp[1], yp[1]],
                    [-x[2], -y[2], -1, 0, 0, 0, x[2]*xp[2], y[2]*xp[2], xp[2]],
                    [0, 0, 0, -x[2], -y[2], -1, x[2]*yp[2], y[2]*yp[2], yp[2]],
                    [-x[3], -y[3], -1, 0, 0, 0, x[3]*xp[3], y[3]*xp[3], xp[3]],
                    [0, 0, 0, -x[3], -y[3], -1, x[3]*yp[3], y[3]*yp[3], yp[3]] ])

    evl_AtA, V = np.linalg.eig(np.dot(np.transpose(A),A))

    H = np.reshape(V[:,-1],[3,3])

    # Z = np.dot(A,H)
    # print(Z)
    # print(Hs)

    return H

# MAIN
lena = cv2.imread('Lena.png')
marker = cv2.imread('ref_marker.png')
marker_grid = cv2.imread('ref_marker_grid.png')

lena_mat, marker_mat = gen_mats()
cap = cv2.VideoCapture('Tag0.mp4')

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
    # print(len(cnt_area))
    # large_cnts = np.zeros(np.shape(mask))
    fresh_im = np.zeros(np.shape(img_orig))
    final_cnts = []
    if len(cnt_area) < 9:
        ## only draw the second largest contour 
        cv2.drawContours(fresh_im, cnts, cnt_num[len(cnt_num)-1-1], (255, 255, 255), -1)
        c = cnts[cnt_num[len(cnt_num)-1-1]]

        mask = cv2.bitwise_and(frame, frame, mask = np.uint8(fresh_im))
        
        epsilon = 0.1*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)
     
        if approx.shape[0] == 4: 
            final_cnts.append(approx)
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
            final_cnts.append(approx)
            final_cnts.append(approx1)
            final_cnts.append(approx2)
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
    
    result = tag_image(frame,final_cnts,marker_mat.copy())

    # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 600,600)
    # cv2.imshow('image',mask)

    cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image1', 600,600)
    cv2.imshow('image1',result)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

