import numpy as np 
import cv2 as cv  

# Remember when we did for lecture 2 the sigma increase and hence the successively blurred image
# That is the same concept here.
# In sift operators the major work is done by Laplacian of Gaussian operator
# Or rather an approximation of the LoG filter
# That will get us edges in an image at different scales.

def main():
    im = cv.imread("balloon.jpg")
    gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)

    # sift = cv.xfeatures2d.SIFT_create()
    # kp = sift.detect(gray,None)

    upscaled = cv.pyrUp(gray)
    down = cv.pyrDown(gray)
    cv.imshow("Upscaled: ", upscaled)
    cv.imshow("Downscaled: ", down)
    cv.imshow("original: ", gray)

    cv.waitKey(0)
    

def scaleSpaceKeyPointDetector():
    im = cv.imread("balloon.jpg")
    gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)

    # Remember that the Laplacian of Gaussian is the double derivitive of the gaussian of an image i.e laplacian(gaussian(image))
    # To make the LoG filter you can also apply laplacian to the gaussian filter as well.

    # You can see that in the three scale the highest response is received by the LoG filter with the lowest sigma value that is because 
    # the laplacian receives very little zero crossings in the Gaussians of the original image with high sigma values
    # This also means that the responses are very little in the other two scales of the laplacian
    # For sift features we take for every pixel we take the keep the scale (i.e. the sigma value) for which the LoG filter gives the highest response
    # At this point we dont care if the response is from an edge or from a corner
    # This is because when we are localising the response we will remove responses which are from edges by using the 2x2 hessian matrix
    # and principal curvature. i.e. very similar to the harris corner detector we will test if the given point is an edge by using the ratio of the 
    # eigen values at the window

    gauss = cv.GaussianBlur(im, (11,11), 1)
    gauss1 = cv.GaussianBlur(im, (11,11), 2)
    gauss2 = cv.GaussianBlur(im, (11,11), 3)
    lap = cv.Laplacian(gauss, -1)
    lap1 = cv.Laplacian(gauss1, -1)
    lap2 = cv.Laplacian(gauss2, -1)
    cv.imshow("Lap scale: 1", cv.dilate(lap, None))
    cv.imshow("Lap scale: 2", cv.dilate(lap1, None))
    cv.imshow("Lap scale: 3", cv.dilate(lap2, None))
    cv.waitKey(0)


if __name__ == "__main__":
    # main()
    scaleSpaceKeyPointDetector();