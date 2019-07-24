import numpy as np
import cv2 as cv    
import sys

def main():
    im = cv.imread("flag.png")
    gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    _, thresh = cv.threshold(gray, 127, 255, 0)

    # Now find contours using the chain approx none parameter

    _, contours, _h = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    cv.drawContours(im, contours[0], -1, (0, 255, 0), 2)
    print("Contours: ", contours)
    print("Contours count: ", len(contours))

    # [Explanation]
    # So the find contorus function works so that the number of areas we have
    # that are contoured seperately, the points for that particular box/shape
    # is stored as a different element on the contours list

    # [Working Example] This example works for the flag.png image
    # print(contours[0][0])    
    # print(contours[1][0])
    # print(contours[2])

    # Lets determine the moments of the given image using the given contours
    # [Working Example] This example works for the flag.png
    # I have arbitrarily chosen the 0th contour and will be working with that for this example
    star = contours[0]
    moments = cv.moments(star) #its important to note that moments are a dictionary that store the value for m00 and m11 and so on till a third degree

    # [Explanation] I would also like to mention the use of moments and the intuition behind there use here
    # So a moment of a 2d function would be, intuitively speaking
    # For m10,we can image that as we go across a particular row of the image
    # the value for our x will keep increase i.e. the idx of our row
    # As we find increments there the weight assigned to the function impulse at
    # that point is greater hence the moment, considering the pivot at the 0th idx
    # is also much greater as we move vertically across the image. 
    # What this does for m10 for example is it finds a summation of
    # the complete row weighted in a linear fashion with farther impulses getting weighted a linearly larger value
    # We also get a larger value exactly for this reason
    # Im also wondering if the contours have to be closed in any way. For that i should
    # probably research how that chain_approximation works

    # Calculating centroids and displaying area for all contours
        # Calculating the moment for all contours
    all_moments = []
    for c in contours:
        all_moments.append(cv.moments(c))

    print(all_moments)
    mom1 = all_moments[0]
    centroidx = mom1['m10']/mom1['m00']
    centroidy = mom1['m01']/mom1['m00']

    cv.drawMarker(im, (int(centroidx), int(centroidy)), (255, 100, 0), cv.MARKER_STAR, 3)

    # Lets also try to bound the star in a bounding box using its contours
    x, y, w, h = cv.boundingRect(contours[0])
    cv.rectangle(im, (x,y), (x+w, y+h), (0, 0, 255), 5)
    box = cv.minAreaRect(contours[0])
    box_p = cv.boxPoints(box)
    box_p = np.int0(box_p)

    cv.drawContours(im, [box_p], 0, (0,255, 255), 2)

    # Lets also label the smallest contour area
    # The area is simply m00

    cv.namedWindow('original', cv.WINDOW_AUTOSIZE)
    cv.moveWindow('original', 0, 0)
    cv.imshow('original',im)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return

if __name__ == "__main__":
    main()