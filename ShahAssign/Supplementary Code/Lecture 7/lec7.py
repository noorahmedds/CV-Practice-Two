import numpy as np 
import cv2 as cv 


# Lets calculate the optical flow here
# So lets try to implement simple optical flow using Horn and Schunk just because it requires me to know some linear algebra 
# syntax so im gonna prepare myself here with that

# So our optical flow equations are (after intense derivation and solving)
# u = uav - fx(P/D)
# v = vav - fy(P/D)
# P = fx*uav + fy*uav + ft
# D = lambda + (fx)^2 + (fy)^2

# uav is basically the avergae of the immediate neightbours of the current u and v
# vav is same
# fx = roberts_x(first im) + roberts_x(second im)
# fy is same
# ft = [-1, -1; -1, -1]*(first_im) + [1, 1; 1, 1]*(second im)

def horn_schunk(seq1, seq2, fx, fy, ft, l):
    

def main():
    # lets first intialize our global varibales
    seq1 = cv.imread("rseq01.jpg")
    seq2 = cv.imread("rseq02.jpg")
    seq1 = cv.cvtColor(seq1, cv.COLOR_RGB2GRAY)
    seq2 = cv.cvtColor(seq2, cv.COLOR_RGB2GRAY)

    roberts_x = np.array([[1, 1],
                          [-1, -1]])
    roberts_y = np.array([[1,-1], 
                          [1,-1]])

    dt_kern_1 = np.ones((2,2))
    dt_kern_2 = -1*np.ones((2,2))

    fx = cv.filter2D(seq1, -1, roberts_x) + cv.filter2D(seq2, -1, roberts_x)
    fy = cv.filter2D(seq1, -1, roberts_y) + cv.filter2D(seq2, -1, roberts_y)
    ft = cv.filter2D(seq1, -1, dt_kern_1) + cv.filter2D(seq2, -1, dt_kern_2)



    cv.imshow("seq1", ft)
    cv.imshow("seq2", fy)

    cv.waitKey(0)


if __name__ == "__main__":
    main()