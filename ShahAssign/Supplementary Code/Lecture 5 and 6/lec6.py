import numpy as np 
import cv2 as cv  

def main():
    im = cv.imread("balloon.jpg")
    gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)

    # sift = cv.xfeatures2d.SIFT_create()
    # kp = sift.detect(gray,None)

    upscaled = cv.pyrUp(gray)
    cv.imshow("Upscaled: ", upscaled)
    cv.imshow("original: ", gray)

    cv.waitKey(0)
    

if __name__ == "__main__":
    main()