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
        
        


def composite_signal():
    pass







