import numpy as np
import cv2 as cv

# So what we can expect to do is create a binary video
# count blobs in a direction. Maybe we can also use some rudimentary form of optical flow
# If we get vectors passing a detection area with an optical flow in the opposite direction we count that as one
# But we need to make sure that we need some form of understanding and memory of a paritcular blob that enters the detection area

def conotouring(frame):
    o_frame = frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,cv.THRESH_OTSU,255,0) 
    frame, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    c_frame = cv.drawContours(o_frame, contours, -1, (0, 255, 0), 3)
    return c_frame
    

def main():
    cap = cv.VideoCapture(0)

    ret = True
    while ret:
        ret, frame = cap.read()

        # do what you want
        frame = conotouring(frame)
        
        cv.imshow("webcam feed", frame)

        waitKey = cv.waitKey(1)
        if waitKey & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()

