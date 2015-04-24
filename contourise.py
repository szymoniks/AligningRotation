
import cv2
import numpy as np
from matplotlib import pyplot as plt

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def show(img):
    plt.imshow(img,cmap = 'gray')
    plt.show()
def match(img1,img2):
    #img1 = cv2.imread(img1pth,0)          # queryImage
    #img2 = cv2.imread(img2pth,0) # trainImage

    # Initiate SIFT detector
    orb = cv2.ORB()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    drawMatches(img1,kp1,img2,kp2,matches[:20])

    #plt.imshow(img3),plt.show()
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
    height, width, depth = img.shape
    empty = np.zeros((height,width,3), np.uint8)
    empty[::] = (255,255,255)    
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
impath = 'test/0x0.png' #master
impath2 = '0.25x1.0.png'
img = cv2.imread(impath) 
img2 = cv2.imread(impath2) 
#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#invert the images
cv2.bitwise_not(img, img)
cv2.bitwise_not(img2, img2)

contoured1 = contourise(img)
show(contoured1)
#contoured2 = contourise(img2)
#contoured1 = cv2.cvtColor(contoured1,cv2.COLOR_BGR2GRAY)
#contoured2 = cv2.cvtColor(contoured2,cv2.COLOR_BGR2GRAY)
#match(contoured1, contoured2)
#orb(contoured)
#plt.imshow(contourise(img),cmap = 'gray')
#plt.show()
