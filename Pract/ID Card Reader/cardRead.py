import cv2 as cv 
import numpy as np 

def detectingBarCodeArea(im):
    """Returns cropped Area"""
    cv.imshow('Original Image', im)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return

def main(fh):
    im = cv.imread(fh)
    detectingBarCodeArea(im)



if __name__ == "__main__":
    fileHandle = "can.jpg"
    main(fileHandle)    