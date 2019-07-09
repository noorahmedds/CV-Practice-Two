import numpy as np 
import cv2 as cv  
import matplotlib.pyplot as plt

def SSD(I0, I1):
    """Finds the SSD between I0 and I1 where I0 is the shifted window and I1 is the image window at x,y"""
    if I0.shape == I1.shape:
        return np.sum(np.square(I0 - I1))
    else: return 0


def generateAutoCorrelationSurface(im, x, y, kernel_size, u_lim, v_lim):
    """ 
        u_lim and v_lim should be half of the actual width and height of the surface plot
        x and y are the centre pixel of the location of the window in the kernel
        kernel size is the the size of the window you are using from the given image. Assuming an odd size greater than 0 and that width and height of the kernel would be same
    """
    # the auto correlation surface is basically the SSD(I(x+u, y+v),I(x, y))
    # The SSD is calculated by subtracting each pixel of the two windows with eachother
    # Square this difference for every pixel
    # Sum the these squared differecnes
    k = kernel_size//2
    I1 = im[x-k:x+k+1, y-k:y+k+1]

    surface = np.zeros((u_lim*2, v_lim*2))

    # Now for u, v we traverse for u, v and find ssd and store in a (u,v) sized array
    for u in range(-u_lim, u_lim):
        for v in range(-v_lim, v_lim):
            I0 = im[x-k+u:x+k+u+1, y-k+v:y+k+v+1]
            surface[u+u_lim, v+v_lim] = SSD(I0, I1)
    
    cv.imshow("Surface", surface)

def harrisCornerDetector(im):
    # What is the aperture parameter for the sobel operator?
    # image, size of the neighbourhood you want to look at so basically the window size
    gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    dst = cv.cornerHarris(gray, 5, 3, 0.04)
    dst2 = cv.cornerHarris(gray, 2, 3, 0.04)

    dst = cv.dilate(dst, None)
    # dst = cv.convertScaleAbs(dst)
    im[dst>0.01*dst.max()]=[0,0,255]
    im[dst2>0.01*dst2.max()]=[0,255, 0]

    print("dst_max: ", np.max(dst))
    print("dst_mean: ", np.mean(dst))
    print("dst_mean: ", np.median(dst))

    cv.imshow("dst", im)
    # cv.imshow("dst2", im)
    cv.imshow("edge", cv.Sobel(im, -1, 1, 1))

    cv.waitKey(0)



if __name__ == "__main__":
    # harrisCornerDetector()

    im = cv.imread("balloon.jpg")
    cv.imshow("original", im)
    # # lets use 205, 131
    # generateAutoCorrelationSurface(im, 205, 131, 11, 10, 10)
    # cv.waitKey(0)

    harrisCornerDetector(im)