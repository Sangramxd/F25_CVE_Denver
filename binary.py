import cv2
import argparse
import os
import numpy as np

#get user inputs
filename = input('Enter filename: ')

#split filename into name and type of image (e.g jpg, png)
name, type = os.path.splitext(filename)

#read original image 
img = cv2.imread(filename)

#gamma correction 
window = 'gamma-corrected'
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 1000, 1000)

def gamma_adjust(x):
    global gamma
    gamma = cv2.getTrackbarPos('gamma', window) / 100
    img_np = np.array(img, dtype=np.float32)
    brightened = 255 * (img_np / 255) ** gamma
    brightened = brightened.astype(np.uint8)

    global gray
    gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)
    cv2.imshow(window, brightened)

#creating each slider
cv2.createTrackbar('gamma', window, 0, 100, gamma_adjust)

gamma_adjust(0)
cv2.waitKey(0)
cv2.destroyAllWindows()

#apply blackhat 
window = 'blackhat-corrected'
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 1000, 1000)

def blackhat_adjust(x):
    global kernel_size
    kernel_size = cv2.getTrackbarPos('kernel_size', window)

    if (kernel_size % 2) == 0: kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    global gray_blackhat
    gray_blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    gray_blackhat = cv2.bitwise_not(gray_blackhat)

    cv2.imshow(window, gray_blackhat)

cv2.createTrackbar('kernel_size', window, 3, 99, blackhat_adjust)

blackhat_adjust(0)
cv2.waitKey(0)
cv2.destroyAllWindows()

#apply binarization
window = 'binary'
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 1000, 1000)

def bin_adjust(x):
    global thresh
    thresh = cv2.getTrackbarPos('thresh', window)

    global bin
    ret, bin = cv2.threshold(gray_blackhat, thresh, 255, cv2.THRESH_BINARY)

    cv2.imshow(window, bin)

cv2.createTrackbar('thresh', window, 0, 255, bin_adjust)

bin_adjust(0)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("%s-bin-%f-%d-%d%s" % (name, gamma, kernel_size, thresh, type), bin)