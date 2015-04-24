
import cv2
import numpy as np
from matplotlib import pyplot as plt

def orb(img):
    #img = cv2.imread(img_pth,0)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Initiate STAR detector
    orb = cv2.ORB()

    # find the keypoints with ORB
    kp = orb.detect(img,None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img,kp[:20],color=(0,255,0), flags=0)
    plt.imshow(img2),plt.show()

def contourise(image):
    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    height = 1080
    width = 1920
    empty = np.zeros((1080,1920,3), np.uint8)
    empty[:,0:0.5*width] = (255,255,255)
    empty[:,0.5*width:width] = (255,255,255)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(empty,contours,-1,(0,255,0),3)
    return empty
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#kernel = np.ones((5,5),np.uint8)

#grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

#cv2.imshow(grad, "img")

#plt.imshow(contours,cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
impath = 'test/0.75x0.5.png'
img = cv2.imread(impath) 
contoured = contourise(img)

orb(contoured)
#plt.imshow(contourise(img),cmap = 'gray')
#plt.show()
