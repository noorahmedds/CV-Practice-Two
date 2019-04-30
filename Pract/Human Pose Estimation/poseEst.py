import numpy as np
import cv2 as cv

# Lets keep this idea of confidence maps
# We are probably going to need to use some form of deep learning

# First lets go over the problem:
# Real-time human pose detection on videos
# Let's break this down into it's components

# General Solution #1
# For the solution to be real time we need some consistent good performace from our algorithm
# This should also depend on what the scenes are composed of
# So in the case where we can not guarantee the amount of humans in a video: We should ensure that our algorithms running time is invariant or atleast least varying when more people are in the video

# General Observation #1
# Baed on a few dozen videos watched on youtube I can see that there is jitter in estimated poses from frame to frame
# I believe in the case that we have a still camera and a moving subject it is very easy to overcome this issue though maintaining temporal information and penalizing estimations which are impossible for the subject to make
# By storing some vector position for a joint we can determine if an estimated pose is even possible compared to the last frame's pose.

# General Solution #2
# Why we should look at HOG's as potential descriptors. Histograms usually bring with them spatial invariance. Which basically means that
# histograms store data which is invariant to the spatial positioning of the data elements stored inside the histogram

# Lets first go through szeliski's book's chapter on this thing
# Notes: 


# Lets go over older papers