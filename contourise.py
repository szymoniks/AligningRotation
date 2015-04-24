
import cv2
import numpy as np
from matplotlib import pyplot as plt

imgray = cv2.imread('chair12.obj.png')
imgray = cv2.cvtColor(imgray,cv2.COLOR_BGR2GRAY)
height = 1080
width = 1920
empty = np.zeros((1080,1920,3), np.uint8)
empty[:,0:0.5*width] = (255,255,255)
empty[:,0.5*width:width] = (255,255,255)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#kernel = np.ones((5,5),np.uint8)

#grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

#cv2.imshow(grad, "img")

#plt.imshow(contours,cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])


cv2.drawContours(empty,contours,-1,(0,255,0),3)
plt.imshow(empty,cmap = 'gray')
plt.show()
