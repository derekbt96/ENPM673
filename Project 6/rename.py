import cv2
import numpy as np
import os
import math

def main():
    snc = 0
    snd = 12500
    PATH = './train'
    # for file in sorted(os.listdir(PATH)):
    #     filepath = os.path.join(PATH,file)
    #     if filepath.endswith(".jpg") or filepath.endswith(".pgm") or filepath.endswith(".png") or filepath.endswith(".ppm"):
    for subdir, dirs, files in os.walk('./train'):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".jpg") or filepath.endswith(".pgm") or filepath.endswith(".png") or filepath.endswith(".ppm"):
                # print(filepath)
                    img = cv2.imread(filepath)
                    img1 = img.copy()

                    if "cat" in filepath:
                        print("Filepath {} has cat".format(filepath))
                        sss = "./train_cats/{0}.png".format(snc)
                        cv2.imwrite(sss,img)
                        snc = snc+1

                    elif "dog" in filepath:
                        print("Filepath {} has dog".format(filepath))
                        sss = "./train_dogs/{0}.png".format(snd)
                        cv2.imwrite(sss,img)
                        snd = snd+1


    print(snc, snd)
                # m = (1,1,1) 
                # s = (1,1,1)
                # img = cv2.randn(img,m,s)
                
                # row,col,ch= img.shape
                # mean = 10
                # var = 100
                # sigma = math.pow(var,0.5)
                # gauss = np.random.normal(mean,sigma,(row,col,ch))
                # gauss = gauss.reshape(row,col,ch)
                # img = img + gauss
                # sss = "./train_renamed/{0}.png".format(sno)
                # cv2.imwrite(sss,img)
                # sno = sno+1


                # row,col,ch = img.shape
                # gauss = np.random.randn(row,col,ch)
                # gauss = gauss.reshape(row,col,ch)        
                # img = img + img * gauss

                # cv2.imshow('Window',img)
                # k = cv2.waitKey(0) & 0xFF
                    
                # if k == 27:      
                #     cv2.destroyAllWindows()
                # if k == ord('e'):
                #     sss = "/home/vdorbala/ICRA/Images/Easy/{0}.png".format(sne)
                #     cv2.imwrite(sss,img)
                #     sne = sne+1
                # if k == ord('m'):
                #     sss = "/home/vdorbala/ICRA/Images/Modest/{0}.png".format(snm)
                #     cv2.imwrite(sss,img)
                #     snm = snm+1
                # if k == ord('h'):
                #     sss = "/home/vdorbala/ICRA/Images/Hard/{0}.png".format(snh)
                #     cv2.imwrite(sss,img)
                #     snh = snh+1



                    # if k == ord('s'):

if __name__ == '__main__':
    main()