import numpy as np 
import cv2 as cv  
import matplotlib.pyplot as plt

def LoG():
    # Laplacian followed by a gaussian. Im going to first just apply the gaussian 
    # First lets show the final image
    im = cv.imread("buildingGray.jpg")
    gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)

    # The gaussian kernel dictates the size of the final kernel aswell
    gauss = cv.GaussianBlur(gray, (7,7), 1)
    lap = cv.Laplacian(gauss, -1)

    cv.imshow("original", im)
    cv.imshow("laplacian", lap)
    cv.imshow("gauss", gauss)

    # now lets do the interesting bit
    # We could also first find the laplacian of the gaussian filter due to associativity
    # Gaussian kernel
    gkern = createGaussianKernel(7,7,1)
    lapkern = cv.Laplacian(gkern, -1)
    lap2 = cv.filter2D(gray, -1, lapkern)

    # notice the shape of the laplacian kernel
    print(lapkern)

    # 255/(2*max) 0.5 shift
    # Need to rescale the final image to match the previous result. But the kernel shape should be enough validation that this works
    cv.imshow("lap2", lap2)

    cv.waitKey(0)



if __name__ == "__main__":
    LoG()