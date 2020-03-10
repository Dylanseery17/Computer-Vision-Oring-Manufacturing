import cv2 as cv
import numpy as np
import time
import math 
from math import pi
import matplotlib.pyplot as plt
import queue 

# Comp Vision Assignment 1 2020 
# Dylan Seery - B00098463

# Gets the amount of pixels within the image
def findMean(img):
    pixels = 0
    amount_of_pixels = 0
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            pixels += img[i,j]
            amount_of_pixels += 1
    return int(pixels/amount_of_pixels)

# Implementation of the clustering algorithm
def sementation(img , average):
    newthresh = average
    oldthreshold = 0
    while abs(newthresh - oldthreshold) > 0.001:
        oldthreshold = newthresh
        y = 0
        x = 0
        amount_of_x = 0
        amount_of_y = 0
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if img[i,j] > newthresh:
                    x += img[i,j]
                    amount_of_x += 1
                if img[i,j] <= newthresh:
                    y += img[i,j]
                    amount_of_y += 1
        newthresh = (x/amount_of_x + y/amount_of_y)/2
    return int(newthresh)

# Applying the threshold value provided by the clustering algorithm
def threshold(img,thresh):
    copy = img.copy()
    for i in range(0, copy.shape[0]):
        for j in range(0, copy.shape[1]):
            if copy[i,j] > thresh:
                img[i,j] = 255
            else:
                img[i,j] = 0
    return img

# Erosion expanding on whitespace within the oring
def erosion(img):
    copy = img.copy()
    for i in range(0, copy.shape[0]):
        for j in range(0, copy.shape[1]):
            try:
                if copy[i,j] == 0:
                    # CHECKING STRUCTURE 
                    for k in range(-1,2):
                        for n in range(-1,2):
                            if copy[i+k,j+n] == 255 and copy[i+k+1,j+n] == 255 and copy[i+k-1,j+n] == 255 and copy[i+k-1,j+n+1] == 255 and copy[i+k-1,j+n-1] == 255:
                                img[i,j] = 255
            except IndexError:
                break
    return img

# Dilation closing up whitespace within the oring
def dilation(img):
    copy = img.copy()
    for i in range(0, copy.shape[0]):
        for j in range(0, copy.shape[1]):
            try:
                if copy[i,j] == 0:
                    # CHECKING STRUCTURE 
                    for k in range(-1,2):
                        for n in range(-1,2):
                            if copy[i+k,j+n] == 255:
                                img[i+1,j+1] = 0
                                img[i-1,j-1] = 0
                                img[i+1,j-1] = 0
                                img[i-1,j+1] = 0
                                img[i-1,j] = 0
                                img[i+1,j] = 0
                                img[i,j+1] = 0
                                img[i,j-1] = 0
            except IndexError:
                break
    return img

# Function for applying both erosion and dilation
def closing(img):
    e = erosion(img)
    d = dilation(e)
    return d

# Connected Component Labelling algorithm  
def component_labelling(img):
    copy = img.copy()
    currlab = 1
    qu = queue.Queue()
    for i in range(0, copy.shape[0]):
        for j in range(0, copy.shape[1]):
            copy[i,j] = 0

    for i in range(0, copy.shape[0]):
        for j in range(0, copy.shape[1]):
            try:
                if copy[i,j] == 0 and img[i,j] == 0:
                    copy[i,j] = currlab
                    img[i,j] = currlab
                    qu.put([i,j])
                    while qu.qsize() != 0:
                        current = qu.get()
                        # CHECKING STRUCTURE
                        structure = [
                            [current[0]+1 , current[1]],
                            [current[0]-1 , current[1]],
                            [current[0] , current[1]+1],
                            [current[0] , current[1]-1],
                        ]
                        
                        # Applying to queue
                        for structure in structure:
                            if img[structure[0], structure[1]] == 0 and copy[structure[0], structure[1]] == 0 :
                                copy[structure[0], structure[1]] = currlab
                                qu.put(structure)
                    # Next object in image
                    currlab += 1
            except IndexError:
                break
    return copy

# Getting center point of image average of all black pixels from connected component labeling
def getCenter(img):
    row_i = 0
    row_j = 0
    counter = 0
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i,j] == 1:
                row_i += i
                row_j += j
                counter += 1
    return [round(row_i/counter), round(row_j/counter)]

# Since we have center point it is possible to get radius
def getRadius(img , center):
    outside_val = 0
    inside_val = 1
    for i in range(0, img[center[0]].shape[0]):
        if img[center[0],i] == 1:
            # moving outter radius more in the oring  
            outside_val = i-1
    for i in range(0, img[center[0]].shape[0]):
        if img[center[0],i] == 0 and img[center[0],i-1] == 1:
            # moving inner radius more in the oring
            inside_val = i-3
            break
    # Distance between 2 lines
    distance = math.hypot(center[0]-center[0], center[1]-outside_val)
    distance1 = math.hypot(center[0]-center[0], center[1]-inside_val)
    return [distance,distance1]

# Final analysising not the best methodology checks within the inner and outter radius for white
# pixels fails image 4 which is incorrect
def analysis(img , center, radius):
    copy = img.copy()
    counter_black = 0
    counter_white = 0
    counter = 0
    result = False
    for i in range(0, copy.shape[0]):
        for j in range(0, copy.shape[1]):
            counter += 1
            try:
                if ((i-center[0])**2 + (j - center[1])**2 < radius[0]**2):
                    if ((i-center[0])**2 + (j - center[1])**2 > radius[1]**2):
                        if copy[i,j] == 1:
                            counter_black+=1
                        elif copy[i,j] == 0:
                            counter_white+=1
            except IndexError:
                break
    defected =  round(counter_white/counter_black*100,2)
    if defected <= 0:
        result = True
    else:
        result = False
    return result
    
# Running program for each image
for x in range(1,16):
    # timer
    start = time.time()
    img = cv.imread('./Oring/Oring'+str(x)+'.jpg',0)

    # Function calls
    thresMean = findMean(img)
    newthresh = sementation(img,thresMean)
    thresh = threshold(img,newthresh)
    close = closing(thresh)
    labeling = component_labelling(close)  
    center = getCenter(labeling)
    radius = getRadius(labeling , center)
    result = analysis(labeling , center, radius)

    # Image back to rgb 
    backtorgb = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
    # Displaying radius
    cv.circle(backtorgb,(center[1], center[0]), int(radius[0]), [0, 0, 255] , 1)
    cv.circle(backtorgb,(center[1], center[0]), int(radius[1]), [0, 255, 0] , 1)
    # IF RESULT TRUE OR FALSE / PASS OR FAIL
    if result:
        cv.putText(backtorgb, 'PASS', (10,215), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    else: 
        cv.putText(backtorgb, 'FAIL', (10,215), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    end = round(time.time()-start , 2)
    # Displaying time
    cv.putText(backtorgb, str(end) + ' seconds', (10,10), cv.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 1, cv.LINE_AA)
    cv.imshow('Image '+str(x)+'',backtorgb)
    cv.waitKey(0)
    cv.destroyAllWindows()