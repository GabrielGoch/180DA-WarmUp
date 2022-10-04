import numpy as np
import cv2 as cv

img = cv.imread('Star.jpeg',0)
ret,thresh = cv.threshold(img,127,255,0)
contours,hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]
x,y,w,h = cv.boundingRect(cnt)
rect = cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv.imshow('rect',rect)
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(img,[box],0,(0,0,255),2)
cv.imshow('rect',img)