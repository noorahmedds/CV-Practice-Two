import numpy as np 
import cv2 as cv  

def driver():
    im = cv.imread('balloon.jpg')
    cv.imshow("balloon", im)


if __name__ == "__main__":
    driver()