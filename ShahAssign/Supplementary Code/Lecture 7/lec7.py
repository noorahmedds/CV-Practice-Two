import numpy as np 
import cv2 as cv 
# import matplotlib.pyplot as plt

# Lets calculate the optical flow here
# So lets try to implement simple optical flow using Horn and Schunk just because it requires me to know some linear algebra 
# syntax so im gonna prepare myself here with that

# So our optical flow equations are (after intense derivation and solving)
# u = uav - fx(P/D)
# v = vav - fy(P/D)
# P = fx*uav + fy*vav + ft
# D = lambda + (fx)^2 + (fy)^2

# uav is basically the avergae of the immediate neightbours of the current u and v
# vav is same
# fx = roberts_x(first im) + roberts_x(second im)
# fy is same
# ft = [-1, -1; -1, -1]*(first_im) + [1, 1; 1, 1]*(second im)

def calc_av(u, i, j):
    # this calculates the uaverage given a pixel in u
    # simple array kay andar se nikaal kay add
    sum = 0
    if (i != 0):
        sum += u[i-1, j]

    if (j!= 0):
        sum += u[i, j-1]

    i_lim = u.shape[0]
    j_lim = u.shape[1]

    if (i != i_lim-1):
        sum += u[i+1, j]
    
    if (j != j_lim-1):
        sum += u[i, j+1]

    return sum/4


def getP(u, v, i, j, fx, fy, ft):
    uav = calc_av(u,i,j)
    vav = calc_av(v,i,j)
    return (fx[i, j] * uav) + (fy[i, j] * vav) + ft[i, j]

def getD(i, j, l, fx, fy):
    return l + pow(fx[i, j], 2) + pow(fy[i, j], 2)
 
def horn_schunk(seq1, seq2, l):
    shape = seq1.shape
    u = np.zeros(shape)
    v = np.zeros(shape)

    roberts_x = np.array([[1., 1.],
                          [-1., -1.]])
    roberts_y = np.array([[1.,-1.], 
                          [1.,-1.]])

    dt_kern_1 = np.ones((2,2))
    dt_kern_2 = -1*np.ones((2,2))

    fx = cv.filter2D(seq1, -1, roberts_x) + cv.filter2D(seq2, -1, roberts_x)
    fy = cv.filter2D(seq1, -1, roberts_y) + cv.filter2D(seq2, -1, roberts_y)
    ft = cv.filter2D(seq1, -1, dt_kern_1) + cv.filter2D(seq2, -1, dt_kern_2)

    # now we need to update u and v for all pixels using our equations
    # we also need to do this for some number of iterations
    iterations = 10
    for count in range(iterations):
        u_dash = np.copy(u)
        v_dash = np.copy(v)
        for i in range(shape[0]):
            for j in range(shape[1]):
                # u = uav - fx(P/D)
                # v = vav - fy(P/D)
                P = getP(u_dash, v_dash, i, j, fx, fy, ft)
                D = getD(i, j, l, fx, fy)
                u[i, j] = calc_av(u_dash, i, j) - (fx[i, j]*(P/D))
                v[i, j] = calc_av(v_dash, i, j) - (fy[i, j]*(P/D))
        print(u[10, 20:50]) #Testing this

    # So we get these beautiful picture which show the movement. You can see that we get negative values because
    # are images are moving to the left
    # The brighter the pixel the larger the value of u or v
    cv.imshow("Final u", u)
    cv.imshow("Final v", v)
    cv.waitKey(0)




def main():
    # lets first intialize our global varibales
    seq1 = cv.imread("rseq01.jpg")
    seq2 = cv.imread("rseq02.jpg")
    seq1 = cv.cvtColor(seq1, cv.COLOR_RGB2GRAY)
    seq2 = cv.cvtColor(seq2, cv.COLOR_RGB2GRAY)



    cv.imshow("seq1", seq1)
    # cv.imshow("seq2", fy)

    horn_schunk(seq1, seq2, 1)

    # cv.waitKey(0)


if __name__ == "__main__":
    main()