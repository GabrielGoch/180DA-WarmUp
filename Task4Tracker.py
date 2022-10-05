import numpy as np
import cv2 as cv

##################################################################################
#
#   Gabriel Goch
#   205088483
#
#   Code modified from Changing Colorspaces and Contour Features
#
##################################################################################

cap = cv.VideoCapture(0)

while(True):
    # Capture frame
    ret, frame = cap.read()
    HSV = True
    if HSV == True:
        # Convert BGR to HSV
        frameColor = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        #Range of bottle color for HSV
        lower_blue = np.array([90, 100, 130])
        upper_blue = np.array([105, 255, 255])
    else:
        #Keep color the same
        frameColor = frame
        #Range of bottle color for BGR
        lower_blue = np.array([100, 60, 0])
        upper_blue = np.array([255, 255, 60])

    # Threshold the image to get only bottle color
    mask = cv.inRange(frameColor, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow('frame', frame)

    #Create tracking rectangle
    ret, thresh = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, 1, 2)
    for conts in contours:
        #Filter out noise and small objects
        area = cv.contourArea(conts)
        if area > 8000:
            #Add tracking rectangle
            rect = cv.minAreaRect(conts)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(res, [box], 0, (0, 0, 255), 2)

    #Show tracked result
    cv.imshow('res', res)

    k = cv.waitKey(2) & 0xFF
    if k == 100:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()