import numpy as np
import cv2
from numpy.linalg import inv
from numpy.linalg import norm
# initial generation of matrix filled with tag and lena indices
# This is done to increase speed of finding warped/unwarped
# images after while applying homography to these points 
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

# return transformation mat 
def pose(H):
    # Intrinsics
    K = np.transpose(np.array([[1406.08415449821,0,0],
        [2.20679787308599, 1417.99930662800,0],
        [1014.13643417416, 566.347754321696,1]]))
    
    K_inv = inv(K).astype(float)
    print(K_inv)
    h1 = H[:,0]
    h2 = H[:,1]

    lamda = 2/(norm(np.matmul(K_inv, h1)) + norm(np.matmul(K_inv, h2)))
    # lamda = 1
    # print(lamda)
    # Bcap = np.dot(inv(K), H)

    B = lamda*np.matmul(K_inv, H)

    if (np.linalg.det(B) < 0):
        B = -1*B
    # print(norm(B[:,0]))
    # print(np.rint(np.linalg.det(B)))
    b1 = B[:,0]
    b2 = B[:,1]
    b3 = B[:,2]

    r1 = b1
    r2 = b2

    r3 = np.cross(np.transpose(r1), np.transpose(r2)).reshape(3,1)
    print(r1)
    print(r2)
    print(r3)
    t  = lamda*b3
    # print(t)
    P = np.dot(K,np.concatenate((lamda*r1, lamda*r2, lamda*r3, t), axis=1))

    # P=P/P[2,3]
    # print(P)
    # print(lamda)
    # print(H)
    return P

    # print(P)

def tag_image(img,contours,marker_matx):
    
    tags = []
    contours_tags = []
    for contour in contours:
        tag_img, H = pull_tag(img,contour,marker_matx)
        P = pose(H)
        val = 80
        axis_ = np.transpose(np.float32(
        [[1, 0, 0, 1], [1, val, 0, 1], [val, val, 0, 1], 
        [val, 0, 0, 1], [1, 0, -val, 1], [1, val, -val, 1], 
        [val, val, -val, 1], [val, 0, -val, 1]]))
        new_points = np.dot(P.copy(),axis_)
        new_points = (new_points[0:2,:]/new_points[2,:]).astype(int)
        # print(new_points)
        cv2.line(img, (new_points[0,0], new_points[1,0]), 
            (new_points[0,1], new_points[1,1]), (0,255,0),3)
        cv2.line(img, (new_points[0,1], new_points[1,1]), 
            (new_points[0,2], new_points[1,2]), (0,255,0),3)
        cv2.line(img, (new_points[0,2], new_points[1,2]), 
            (new_points[0,3], new_points[1,3]), (0,255,0),3)
        cv2.line(img, (new_points[0,3], new_points[1,3]), 
            (new_points[0,0], new_points[1,0]), (0,255,0),3)

        cv2.line(img, (new_points[0,4], new_points[1,4]), 
            (new_points[0,5], new_points[1,5]), (0,255,0),3)
        cv2.line(img, (new_points[0,5], new_points[1,5]), 
            (new_points[0,6], new_points[1,6]), (0,255,0),3)
        cv2.line(img, (new_points[0,6], new_points[1,6]), 
            (new_points[0,7], new_points[1,7]), (0,255,0),3)
        cv2.line(img, (new_points[0,7], new_points[1,7]), 
            (new_points[0,4], new_points[1,4]), (0,255,0),3)

        cv2.line(img, (new_points[0,0], new_points[1,0]), 
            (new_points[0,4], new_points[1,4]), (0,255,0),3)

        cv2.line(img, (new_points[0,1], new_points[1,1]), 
            (new_points[0,5], new_points[1,5]), (0,255,0),3)
        cv2.line(img, (new_points[0,2], new_points[1,2]), 
            (new_points[0,6], new_points[1,6]), (0,255,0),3)
        cv2.line(img, (new_points[0,3], new_points[1,3]), 
            (new_points[0,7], new_points[1,7]), (0,255,0),3)
        

        if tag_img is not None:
            # Tag contour in the video frame is appended
            contours_tags.append(contour)
            # Append tag id's 
            tags.append(decode(tag_img))
    
    ## Frames with tag id 
    added_frame = add_tag_imgs(img, contours_tags, tags)
    return added_frame

# returns image of the tag alone
def pull_tag(img,contour_tag,marker_matx):
    # form an 80X80 image of the tag 
    draw = np.zeros([80,80,3],np.uint8)
    # These are the four corner points of the newly formed tag image
    x = np.array([1, 80, 80, 1])
    y = np.array([1, 1, 80, 80])
    # These are the four corner points of the detected contours
    xp = np.array([contour_tag[0,0,0], contour_tag[1,0,0], contour_tag[2,0,0], contour_tag[3,0,0]])
    yp = np.array([contour_tag[0,0,1], contour_tag[1,0,1], contour_tag[2,0,1], contour_tag[3,0,1]])
    # find homography with just these four points
    H = find_homography(x,y,xp,yp)
    # find out the indices of the tag in the video frame
    xy_new = np.dot(H,marker_matx)
    # homogeneous to cartesian coordinate conversion
    xy_new = ((xy_new[0:2,:]/xy_new[2,:])).astype(int) 
    # Check if the new points are valid. i.e. Are they within the img size bounds
    if (xy_new < 0).any() or (xy_new[0,:] > 1920).any() or (xy_new[1,:] > 1080).any():
        return None
    # Populate the tag image once you know the desired indices in video frame
    draw[marker_matx[1,:],marker_matx[0,:]] = img[xy_new[1,:],xy_new[0,:]]
    return draw, H

# returns the id of the tag
def decode(marker_image):
    marker_img = cv2.resize(marker_image, (80, 80), interpolation = cv2.INTER_AREA)
    marker_img = cv2.cvtColor(marker_img,cv2.COLOR_BGR2GRAY)
    # Binary thresh for better contrast
    ret,th1 = cv2.threshold(marker_img,128,255,cv2.THRESH_BINARY)
    # Erosion  
    kernel = np.ones((3,3), np.uint8)  
    th1 = cv2.erode(th1, kernel, iterations=1) 

    # checkout the orientation by summing the pixel values at the 
    # corner points starting from top left (3,3), (3,6), (6,3), (6,6) in the 8X8 grid
    orientation_tags = np.array([np.sum(th1[21:29,21:29]),
                        np.sum(th1[21:29,51:59]),
                        np.sum(th1[51:59,21:29]),
                        np.sum(th1[51:59,51:59])])
    # The middle four data tags
    data_tags = np.array([np.sum(th1[34:37,34:37])/9.0,
                        np.sum(th1[34:37,44:47])/9.0,
                        np.sum(th1[44:47,34:37])/9.0,
                        np.sum(th1[44:47,44:47])/9.0])

    data_tags = (data_tags > 225).astype(int)

    corner = np.argmax(orientation_tags)
    if corner == 3: # The tag's orientation is the same as ground truth 
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

    th_ = cv2.cvtColor(th1,cv2.COLOR_GRAY2BGR)
    x = [10, 20, 30, 40, 50, 60, 70, 80]
    for i in range(len(x)):
        cv2.line(th_, (x[i], 0), (x[i], 80), (0,255,0),1)
        cv2.line(th_, (0, x[i]), (80, x[i]), (0,255,0),1)
    # rotated marker i.e. marker display in ground truth orientation 
    cv2.imshow('marker',th_)
    cv2.waitKey(1)

    return tag_ID

# returns tag id displayed over the image 
def add_tag_imgs(img, contours, tags):

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0) #B
    thickness = 2
    
    indx = 0
    for contour in contours:
        pnt = (max(contour[:,:,0]),min(contour[:,:,1]))
        image = cv2.putText(img, str(tags[indx]), pnt, font, fontScale, color, thickness, cv2.LINE_AA) 
        indx += 1
    return img

# Superimpose lena's image 
def superImpose(dest,contours,Lena,lena_matx):

    for contour in contours:

        x = np.array([1, 512, 512, 1])
        y = np.array([1, 1, 512, 512])
        xp = np.array([contour[0,0,0], contour[1,0,0], contour[2,0,0], contour[3,0,0]])
        yp = np.array([contour[0,0,1], contour[1,0,1], contour[2,0,1], contour[3,0,1]])
        H = find_homography(x,y,xp,yp)

        lena_temp = lena_matx.copy()
        xy_new = np.dot(H,lena_temp)
        xy_new = ((xy_new[0:2,:]/xy_new[2,:])).astype(int)
        if (xy_new < 0).any() or (xy_new[0,:] > 1920).any() or (xy_new[1,:] > 1080).any():
            return None, None
        dest[xy_new[1,:],xy_new[0,:]] = Lena[lena_temp[1,:],lena_temp[0,:]]

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
    H = H/H[2,2]
    return H

# MAIN
lena = cv2.imread('Lena.png')
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
    mask = []
    if len(cnt_area) < 9:
        ## only draw the second largest contour
        for cnt in cnts:
             cv2.drawContours(fresh_im, cnts, cnt_num[len(cnt_num)-1-1], (255, 255, 255), -1)
        
        c = cnts[cnt_num[len(cnt_num)-1-1]]
        # print(cv2.contourArea(c))
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
    result = superImpose(result,final_cnts,lena,lena_mat.copy())

    cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image1', 600,600)
    cv2.imshow('image1',result)

    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600,600)
    cv2.imshow('image',mask)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

