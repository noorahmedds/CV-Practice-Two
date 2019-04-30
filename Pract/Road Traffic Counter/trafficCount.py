import numpy as np
import cv2 as cv

# So what we can expect to do is create a binary video
# count blobs in a direction. Maybe we can also use some rudimentary form of optical flow
# If we get vectors passing a detection area with an optical flow in the opposite direction we count that as one
# But we need to make sure that we need some form of understanding and memory of a paritcular blob that enters the detection area



def main():

    # Lets first setup our bacground subtractor

    # Then we subtract our background to get foreground moving objects

    # Apply filtering to reduce noise for example something like opening which reduces false positive foreground

    # I want to do something like optical flow to get direction of moving vehicles and then detect whether they have entered my mask

    # The problem is retaining the count

    # Future problem: How many cars shifted lanes, how many cars were red

    cap = cv.VideoCapture("input.mp4")

    ret = True
    while ret:
        ret, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow("Video Feed", frame)

        waitKey = cv.waitKey(1)
        if waitKey & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()

