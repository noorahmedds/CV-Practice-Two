import cv2 as cv 
import numpy as np 
import imutils

def detectingBarCodeArea(im):
    """Returns cropped Area"""
    cv.imshow('Original Image', im)

    # So first things first we need to apply this canny edge detector in both directions to get image derivitive.
    # blur the image after this
    # Based on this we need to threshold the image, ill apply otsus threshold
    # After that we need to find segments of the image/contours of the image
    # Open the image so that we can reduce stray blobs or we can just take the biggest contour/blob with the biggest area
    # That will be the image itself.

    ddepth = cv.cv.CV_32F if imutils.is_cv2() else cv.CV_32F
    gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    dx = cv.Sobel(gray, ddepth=ddepth, dx=1 , dy=0, ksize=1)
    dy = cv.Sobel(gray, ddepth=ddepth, dx=0 , dy=1, ksize=1)

    # dxy = dx-dy essentially want to do this. As it will reduce the stray edges of the horizontal gradients vs the vertical edges
    dx_clean = cv.subtract(dx, dy)
    # dx_clean = cv.convertScaleAbs(dx_clean)
    dxy = cv.add(dx, dy)
    # d_canny = cv.Canny(gray, 50, 100)
    
    # lets compare dx
    # cv.imshow("DX", dx)
    # cv.imshow("DX_CLEAN", dx_clean)
    # cv.imshow("DXY", dxy)
    # cv.imshow("Dcanny", d_canny)

    # now we need to blur the image so that while applying threshold we get some good blobs and contours
    gx = cv.GaussianBlur(dx_clean,(3,3), 2)
    # cv.imshow("GX", gx)

    # print(type(gx))

    ret,threshed = cv.threshold(gx,70,255,cv.THRESH_BINARY)
    cv.imshow('Thresholded Im', threshed)

    kernel = np.ones((10,1),np.uint8)
    kernel_close = np.ones((5,10),np.uint8)
    # because we need to preserve long lines we should open with a long kernel. This should technically remove weird noise
    opened = cv.morphologyEx(threshed, cv.MORPH_OPEN, kernel, iterations=1)

    # now that we have a thresholded image, lets open the image to get a better blob
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel_close, iterations=2)
    cv.imshow('Opened Im', opened)
    cv.imshow('Closed Im', closed)

    # To make in comply with the cv.connectedComponents
    closed = cv.convertScaleAbs(closed)
    cv.imshow('Closed Im', closed)

    # ret, comped = cv.connectedComponents(closed)
    # cv.imshow('Comped', comped)


    cnts = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key=cv.contourArea, reverse=True)[0]
    
    rect = cv.minAreaRect(c)
    box = cv.cv.BoxPoints(rect) if imutils.is_cv2() else cv.boxPoints(rect)
    box = np.int0(box)

    cv.drawContours(im, [box], -1, (0,255, 0), thickness=1)
    cv.imshow('finResult', im)




    cv.waitKey(0)
    cv.destroyAllWindows()
    return

def main(fh):
    im = cv.imread(fh)
    detectingBarCodeArea(im)



if __name__ == "__main__":
    fileHandle = "can.jpg"
    main(fileHandle)    