
import cv2, itertools, glob,sys
import numpy as np
from matplotlib import pyplot as plt

def show(img):
    plt.imshow(img,cmap = 'gray')
    plt.show()

def get_cnt(image):
    #imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    ret,thresh = cv2.threshold(image,127,255,0)
    _, contours0, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = np.array([cv2.approxPolyDP(cnt, 3, True) for cnt in contours0])
    return contours

def return_empty_image(h,w):
    empty = np.zeros((h,w,3), np.uint8)
    empty[::] = (255,255,255)
    return empty

def apply_contour(cnt, image):   
    cv2.drawContours(image,cnt,-1,(0,255,0),3)
    return image

def translate_img(img, x, y):
    M = np.matrix([[1,0,x],[0,1,y]])
    cv2.warpAffine(img, M, img.shape[:2]) 
    return img

def translate_cnt(cnt, x, y):
    for poly in cnt:
        for point in poly:
            point[0][0] += x
            point[0][1] += y
    return cnt

def center_cnt(cnt, img):
    cnt_rect = np.array([ item for l in cnt for item in l]) 
    #print "cnts: {}".format(cnt)
    imh,imw = img.shape[:2]
    x,y,w,h = cv2.boundingRect(cnt_rect)
    shiftx = (imw*.5) - (x + (.5*w))
    shifty = (imh*.5) - (y + (.4*h))
    cnt = translate_cnt(cnt, shiftx, shifty)
    return cnt

def get_bound(cnt):
    cnt_rect = np.array([ item for l in cnt for item in l]) 
    x,y,w,h = cv2.boundingRect(cnt_rect)
    return x,y,w,h

def draw_bound(img, cnt):
    x,y,w,h = get_bound(cnt)
    rect = ((x,y),(x+w,y+h))
    cv2.rectangle(img, rect[0], rect[1], (0,0,0))
    return img

def correct(d, cnt):
    _,_, chair_width,_ = get_bound(cnt)
    right_side = (d['tr'],d['br'])
    #69,55,59
    tldiff = (100* abs(d['tl']))/( ( ( abs(d['tr'])+abs(d['br']) )/2.)) - 100
    print "tl is {}% farther than right side".format(tldiff)
    print d
    print chair_width
    print "bottom left - top left: {}".format(abs(d['bl']-d['tl']))
    #bottom left is closer to chair than top left by more than 30% a chair width (30% based  on 0.25x1)
    cond1 = abs(d['bl']) < abs(d['tl']) and (abs(d['bl']-d['tl']) > .3*chair_width)
    #bottom left and bottom right are within half a chair width of each other
    cond2 = abs(max((d['bl'], d['br']))-min((d['bl'], d['br']))) < .5*chair_width
    #the top right and bottom right values are roughly equidistant from the chair
    cond3 = abs(max(right_side)-min(right_side)) < .1*abs(max(right_side))
    
    return cond1 and cond2 and cond3

def get_distances(height, width, cnt):
    # x,y = width, height
    pt_cof = (.85,.1)
    
    checkpoints = itertools.product(pt_cof, repeat=2)
    order = ('br','tr','bl','tl')
    dist = {}
    for i,coef in enumerate(checkpoints):
        pt = (int(width*coef[0]), int(height*coef[1]))
        d = min([cv2.pointPolygonTest(c, pt,True) for c in cnt], key=abs)
        dist[order[i]] = d  

    return dist

def visualiser(img, cnt):
    _img = img.copy()
    height, width, _ = _img.shape
    _img = apply_contour(cnt, _img)
    # x,y = width, height

    pt_cof = (.85,.1)
        
    #goes: bottom right, top right, bottom left, top left
    checkpoints = itertools.product(pt_cof, repeat=2)
   
    for coef in checkpoints:
        pt = (int(width*coef[0]), int(height*coef[1]))
        dist = min([cv2.pointPolygonTest(c, pt,True) for c in cnt], key=abs)
        cv2.putText(_img, str(dist), pt, cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
        cv2.circle(_img, pt, 10, (255, 0, 0))  
    _img = draw_bound(_img, cnt)
    return _img

def main(args):
#usage, python contourise2 <directory of pngs>
    directory = args[0]
    #goes through every image
    candidates = glob.glob(directory+"/*.png")
    #candidates = ["test/0.25x1.0.png"]
    for candidate in candidates:
        cndt = cv2.imread(candidate)
        #invert it
        cv2.bitwise_not(cndt, cndt)
        #grayscale
        cndt = cv2.cvtColor(cndt,cv2.COLOR_BGR2GRAY)
        height, width = cndt.shape[:2]
        #get outline
        cnt = get_cnt(cndt)
        empty = return_empty_image(height,width)  
        #make sure the outline is centered in the empty image
        cnt = center_cnt(cnt, empty)

        #draw the outline onto empty image
        #get distances from reference points to contour
        dists = get_distances(height, width, cnt)

        #finally, check if its the aspect we're looking for
        ismatch = correct(dists,cnt)
        print "match: {}, img: {}\n\n".format(ismatch, candidate)
        #visualise it for fun
        
        show(visualiser(empty, cnt))

if __name__ == "__main__":
    main(sys.argv[1:])
#plt.show()
