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


def yiq_frequency(input):

    cap = cv2.VideoCapture(input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(fps * 3)

    frames = []
    for i in range(num_frames):

        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

    # Extract the Y component
    YIQ = [cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb) for frame in frames]
    
    y_channels = []
    I_channels = []
    Q_channels = []
    
    for frame in YIQ:
        y, i, q = cv2.split(frame)
        y_channels.append(y)
        I_channels.append(i)
        Q_channels.append(q)

    # Perform 2D Fourier Transform

    y_frequency = [np.fft.fft2(y_channel) for y_channel in y_channels]
    
    i_frequency = [np.fft.fft2(i_channel) for i_channel in I_channels]

    q_frequency = [np.fft.fft2(q_channel) for q_channel in Q_channels]


    # Shift zero frequency component to center

    mag_spectrum_Y  = [np.log(np.abs(np.fft.fftshift(fft_frame))+ 1) for fft_frame in y_frequency] # Log-scaled for visualization

    mag_spectrum_I = [np.log(np.abs(np.fft.fftshift(fft_frame))+ 1) for fft_frame in i_frequency]

    mag_spectrum_Q = [np.log(np.abs(np.fft.fftshift(fft_frame))+ 1) for fft_frame in q_frequency]

    # Plot the magnitude spectrum

    for i in range(len(mag_spectrum_Y)):

        # Display magnitude spectrum

        plt.imshow(mag_spectrum_Y[i], cmap='gray')

        # Display magnitude spectrum

        plt.imshow(mag_spectrum_I[i], cmap='gray')

        # Display magnitude spectrum

        plt.imshow(mag_spectrum_Q[i], cmap='gray')
    
        # Wait for a key press.
        cv2.waitKey(0)

        # Destroy the windows.
        cv2.destroyAllWindows()
        
def display_rgb_freq(r, g, b):
    
        mag_r, mag_g, mag_b = rgb_frequency(r, g, b)
        
        mag_r= 20 * np.log10(np.abs(mag_r))
        mag_g = 20 * np.log10(np.abs(mag_g))
        mag_b = 20 * np.log10(np.abs(mag_b))
        
        # Display the magnitude spectrum
        plt.figure(3)
        plt.semilogy(mag_r[::10])
        plt.title('R original in Frequency Domain')
        
        # Display the magnitude spectrum
        plt.figure(4)
        plt.semilogy(mag_g[::10])
        plt.title('G original in Frequency Domain')
     
        
        # Display the magnitude spectrum
        plt.figure(5)
        plt.semilogy(mag_b[::10])
        plt.title('G original in Frequency Domain')

        
        plt.show(block=False)
        
def display_rgb(r, g, b):
    
        # We will now represent each frame in R G B image form
        # Display the R, G, and B frames.
        cv2.imshow("R", r)
        cv2.imshow("G", g)
        cv2.imshow("B", b)
    
        Rs, Gs, Bs = rgb_spatial(r, g, b)
        
        # Plot the result in spatial domain
        plt.figure(0)
        plt.plot(Rs)
        plt.title('R original in Spatial Domain')

        # Plot the result in spatial domain
        plt.figure(1)
        plt.plot(Gs)
        plt.title('G original in Spatial Domain')
        
        # Plot the result in spatial domain
        plt.figure(2)
        plt.plot(Bs)
        plt.title('B original in Spatial Domain')
        
        plt.show(block=False)
      
    
        
def rgb(input):
    cap = get_video(input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(fps * 3)
    
    
    for i in range(num_frames):
        ret, frame = cap.read()
    
        # Check if the frame was read successfully.
        if not ret:
            break
        
        r, g, b = conv_to_rgb(frame)
        
        display_rgb(r, g, b)
        
        # Wait for a key press.
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        display_rgb_freq(r, g, b)
        
        # Wait for a key press.
        key = cv2.waitKey(0)
        
        if key == ord('0'):
            # Destroy all figures
            plt.close('all')
        elif key == ord('q'):
            break

    # Release the video capture object
    cap.release()
    # Destroy any remaining OpenCV windows
    cv2.destroyAllWindows()


def conv_to_rgb(input):
    r, g, b = cv2.split(input)
    
    return r, g, b
    
        
def rgb_spatial(r, g, b):
    
    # Reshape the R channel to a 1D array
    Rspatial = r.flatten()
    Gspatial = g.flatten()
    Bspatial=  b.flatten()
    
    return Rspatial, Gspatial, Bspatial

    
def rgb_frequency(r, g, b):
    
    r, g, b = rgb_spatial(r, g, b)
    
    # Perform 1D FFT
    f_r = np.fft.fft(r)
    f_g = np.fft.fft(g)
    f_b = np.fft.fft(b)

    # Calculate the real parts
    R_frequency_domain = np.real(f_r)
    G_frequency_domain = np.real(f_g)
    B_frequency_domain = np.real(f_b)

    # Take the first half (due to symmetry)
    R_frequency_domain = R_frequency_domain[:len(R_frequency_domain)//2]
    G_frequency_domain = G_frequency_domain[:len(G_frequency_domain)//2]
    B_frequency_domain = B_frequency_domain[:len(B_frequency_domain)//2]

    
    return R_frequency_domain, G_frequency_domain, B_frequency_domain


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
    
    cv2.imshow("c", c_frame)
    
    
    # Wait for a key press.
    cv2.waitKey(0)

    # Destroy the windows.
    cv2.destroyAllWindows()
    
    return frame
    
    








