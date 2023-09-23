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
    cap = get_video(input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(fps * 3)
    
    ret, frame = cap.read()
  
    # Convert the image to YIQ format.
    yiq_image = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    
    y, i, q = cv2.split(yiq_image)
    

    cv2.imshow("Y", y)
    cv2.imshow("I", i)
    cv2.imshow("Q", q)
    
     # Wait for a key press.
    cv2.waitKey(0)

        # Destroy the windows.
    cv2.destroyAllWindows()
    
    
    # Return the YIQ image.
    return yiq_image

def conv_to_rgb(input):
    cap = get_video(input)
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
        
        



def composite_signal(input):
    frame = conv_to_yiq(input)
    
     # Split the frame into its Y, I, and Q components.
    y, i, q = cv2.split(frame)

    # Apply a low-pass filter to the Y component. 4.2Mhz
    kernel_size = int(np.ceil(4 / 4200000))
    kernel = cv2.getGaussianKernel(kernel_size, kernel_size / 4)
    y = cv2.filter2D(y, -1, kernel)

    # Apply a low-pass filter to the I component. 1.5Mhz
    kernel_size = int(np.ceil(4 / 1500000))
    kernel = cv2.getGaussianKernel(kernel_size, kernel_size / 4)
    i = cv2.filter2D(i, -1, kernel)

    # Apply a low-pass filter to the Q component. 0.5 Mhz
    kernel_size = int(np.ceil(4 / 500000))
    kernel = cv2.getGaussianKernel(kernel_size, kernel_size / 4)
    q = cv2.filter2D(q, -1, kernel)
    # Combine the Y, I, and Q components back into a single frame.
    c_frame = cv2.merge((y, i, q))
    
    cv2.imshow("c", y)
    
    
    # Wait for a key press.
    cv2.waitKey(0)

    # Destroy the windows.
    cv2.destroyAllWindows()
    
    return frame
    
    








