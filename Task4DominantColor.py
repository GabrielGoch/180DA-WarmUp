import cv2 as cv
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

##################################################################################
#
#   Gabriel Goch
#   205088483
#
#   Code modified from Changing Colorspaces and Finding Dominant Colour on an Image
#
##################################################################################


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


cap = cv.VideoCapture(0)

while(True):
    # Capture frame
    ret, frame = cap.read()
    croppedFrame = frame[329:389,609:669]

    cv.imshow('frame', frame)

    croppedFrame = croppedFrame.reshape((croppedFrame.shape[0] * croppedFrame.shape[1], 3))  # represent as row*column,channel number
    clt = KMeans(n_clusters=3)  # cluster number
    clt.fit(croppedFrame)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    #Display Result
    cv.imshow('bar',bar)

    k = cv.waitKey(2) & 0xFF
    if k == 100:
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()