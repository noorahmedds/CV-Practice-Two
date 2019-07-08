import numpy as np 
import cv2 as cv  
from scipy import signal
import matplotlib.pyplot as plt

# remember to use better parameter names. In image processing it is very easy to confuse the order of parameters.
# with most libraries opting for the (height width) order it is important to not confuse this with the [i, j] convention
# nor the [x,y] convention. The order isnt same here. Height as in the Y dimension comes first in matrix operations
# or in m x n convention.
# If youre interested in more about gaussian kernels and so on refer to the following: https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm

def createGaussianKernel(height, width, sigma):
    # Lets also learn how we create a gaussian kernel in the first place with a specified width and height and also a sigma value.
    g1 = signal.gaussian(height, sigma).reshape((height, 1))
    g2 = signal.gaussian(width, sigma)
    G = np.outer(g1, g2)
    # this kernel should g1 x g2 
    # derived from the nx1 into 1xm rule. nxm should be the answer
    
    # Simple way to understand this is that we get one fimensional gaussiam distribution with given sigma
    # using the gaussian equation
    # to get two dimesnional gaussian kernel you need to convolute g1 with g2. That should look like an outer product.
    # Because g1 and g2 are both transformations. I believe that applying them individually to images is similar to multiplying them together
    # then applying to image. See the following equation
    # g1 x g2 x im
    # (g1 x g2) x im (because of the associative property)
    # and hence G x im = fin
    return G

def gaussian_vs_mean():
    im = cv.imread('balloon.jpg')
    gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)

    # a simple 3 by 3 gaussian filter applied to our image
    gauss3b3_1 = cv.GaussianBlur(gray, (3,3), 1) #see what border type filtering does
    gauss11b11_1 = cv.GaussianBlur(gray, (11,11), 1) #see what border type filtering does

    cv.imshow("original", gray)
    cv.imshow("gauss3b3_1", gauss3b3_1)
    cv.imshow("gauss11b11_1", gauss11b11_1)
    # You should be able to see the sharpness difference between the three images. 
    # Whats happening here?
    # ans: As we increase the kernel size we get a bigger kernel size and hence the amount of pixels that are weighted
    # each time increases. The middle pixel gets more information from farther neighbours and hence this averages out
    # and blurs relative to a larger portion in the image.
    # Thinking about it mathematically: How can we explain gaussian convolution mathematically:
    # Coming straight from the linear algebra course im trying to understand what this convolution operation does to the image
    # Convolution is basically a dot product between two matrices. I want to say its really a dot product between two vectors of n dimensions
    # which is not true. For our image each pixel is a vector (color/grayscale, locx, locy). But to convert to this gaussian spread image
    # we need to attain this long vector where our vector is (kernel_width * kernel_height) long. The dot product would then be the projection
    # of this vector onto the gaussian space. And each projection would just be an averaging. So we come full circle there.

    # Lets show these two kernels to get a better idea of whats going on here:
    print(createGaussianKernel(3,3,1))
    print(createGaussianKernel(11,11,1))

    # Remember a larger kernel std dev would mean that the weights are spread over a larger area. So lets say that atleast 66 percent of values
    # of the gaussian kernel would be between 10 and -10 pixels of this 50x50 kernel.
    # which would include large numbers. This means that the 10 neighbours of the current pixel would be given atleast some
    # weight and hence averaging the thing further. 
    cv.imshow("Kernel", createGaussianKernel(50,50,10))
    
    cv.imshow("Mean im", cv.filter2D(gray, -1, kernel=np.ones((11,11)) * 1/(11*11)))
    # Compare the 11x11 gaussian image with the mean one. You will see a stark difference. Firstly the gaussian retains major 
    # information of the current pixel and applies a gradient blur. It also retains structure better compared to a mean
    
    cv.waitKey(0)


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
    # gaussian_vs_mean()
    LoG()