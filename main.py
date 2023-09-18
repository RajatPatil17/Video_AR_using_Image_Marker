import cv2
import numpy as np

cap = cv2.VideoCapture(0)
imgtarget = cv2.imread('image.jpg')
myvid = cv2.VideoCapture('video.mp4')
detection = False
framecounter = 0
success, imgvideo = myvid.read()
ht, wt, ct = imgtarget.shape
imgvideo = cv2.resize(imgvideo,(wt,ht))

orb = cv2.ORB_create(nfeatures = 1000)
kp1, des1 = orb.detectAndCompute(imgtarget, None)
imgtarget = cv2.drawKeypoints(imgtarget, kp1, None)

while True:
    success, imgwebcam = cap.read()
    kp2, des2 = orb.detectAndCompute(imgwebcam, None)
    imgaug = imgwebcam.copy()
    if detection == False:
        myvid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        framecounter = 0
    else: 
        if framecounter == myvid.get(cv2.CAP_PROP_FRAME_COUNT):
            myvid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            framecounter = 0
        success, imgvideo = myvid.read()
        
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 *n.distance:
            good.append(m)
    print(len(good))
    imgfeatures = cv2.drawMatches(imgtarget, kp1, imgwebcam, kp2, good, None, flags = 2)
    
    if len(good) > 5:
        detection = True
        srcpts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstpts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        
        matrix, mask = cv2.findHomography(srcpts, dstpts, cv2.RANSAC, 5)
        print(matrix)
        
        pts = np.float32([[0,0],[0,ht], [wt,ht], [wt,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(imgwebcam,[np.int32(dst)], True, (255,0,255), 3)    
        imgwarp = cv2.warpPerspective(imgvideo, matrix, (imgwebcam.shape[1], imgwebcam.shape[0]))
        masknew = np.zeros((imgwebcam.shape[0],imgwebcam.shape[1]),np.uint8)
        cv2.fillPoly(masknew, [np.int32(dst)], (255,255,255))
        maskinv = cv2.bitwise_not(masknew)
        imgaug = cv2.bitwise_and(imgaug, imgaug, mask = maskinv)
        imgaug = cv2.bitwise_or(imgwarp, imgaug)
    cv2.imshow('new mask', imgaug)
    #cv2.imshow('warped', imgwarp)
    #cv2.imshow('img2', img2)
    #cv2.imshow('imgfeatures', imgfeatures)
    #cv2.imshow('Image target', imgtarget)
    #cv2.imshow('video', imgvideo)
    cv2.imshow('webcam', imgwebcam)
    cv2.waitKey(1)
    framecounter+=1