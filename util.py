'''
Ridwan Alrefai - Matteo
University of Illinois Chicago
util.py
   
   
This file contains all opencv and image manipulation functions as utiliy to main.py 
Created for simplicity and ease of use.
   
   
'''


import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import time


# calls opencv video capture
def get_video(input):
    return cv2.VideoCapture(input)
    
    

def conv_to_yiq(input):
    pass

def conv_to_rgb(input):
    cap = cv2.VideoCapture(input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(fps * 3)
    
    frames = []
    
    for i in range(num_frames):
        # Read the frame.
        ret, frame = cap.read()

        # Check if the frame was read successfully.
        if not ret:
            break

        # Split the frame into R, G, and B channels.
        r, g, b = cv2.split(frame)

        # Add the R, G, and B frames to the list.
        frames.append((r, g, b))
        
    
    for i in range(num_frames):
        # Get the R, G, and B frames.
        r, g, b = frames[i]

        # Display the R, G, and B frames.
        cv2.imshow("R", r)
        cv2.imshow("G", g)
        cv2.imshow("B", b)

        # Wait for a key press.
        cv2.waitKey(0)

        # Destroy the windows.
        cv2.destroyAllWindows()
        
        
        
def rgb_frequency(input):
    cap = cv2.VideoCapture(input)
    
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
    #rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    
    grey_scale = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    
    
    fft = [np.fft.fft2(grey) for grey in grey_scale]
    
    mag_spectrum = [np.log(np.abs(np.fft.fftshift(fft_frame)) + 1) for fft_frame in fft]
    
    for mag in mag_spectrum:
        # Display magnitude spectrum
        plt.imshow(mag, cmap='gray')
        plt.title('FFT of Grayscale Frame'), plt.xticks([]), plt.yticks([])
        plt.show()
        
        # Wait for a key press.
        #cv2.waitKey(0)
        #cap.release()

        # Destroy the windows.
        cv2.destroyAllWindows()
        
        



def composite_signal():
    pass







