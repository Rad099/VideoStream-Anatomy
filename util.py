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

plt.rcParams['agg.path.chunksize'] = 1000


composite_frames = []
rec_yiq_frames = []
rec_rgb_frames = []


##### GLOBAL VARIABLES #####
max_Frequency = 5450000 # as shown in slides, not including audio signal. NTSC standard frequency ranges
Y_carrier_Frequency = 1250000
I_carrier_Frequency = 3875000
Q_carrier_Frequency = 4830000
I_beginning_Frequency = 3398001
Q_beginning_Frequency = 4352001


# calls opencv video capture
def get_video(input):
    return cv2.VideoCapture(input)

     
def conv_to_yiq(input):
    
    yiq_image = cv2.cvtColor(input, cv2.COLOR_RGB2YCrCb)

    y, i, q = cv2.split(yiq_image)

    
    # Return the YIQ image.
    return y, i, q

def display_yiq(y, i, q):
    # We will now represent each frame in R G B image form
        # Display the R, G, and B frames.
        cv2.imshow("Y", y)
        cv2.imshow("I", i)
        cv2.imshow("Q", q)
    
        Ys, Is, Qs = yiq_spatial(y, i, q)
        
        # Plot the result in spatial domain
        plt.figure(0)
        plt.plot(Ys)
        plt.title('Y original in Spatial Domain')

        # Plot the result in spatial domain
        plt.figure(1)
        plt.plot(Is)
        plt.title('I original in Spatial Domain')
        
        # Plot the result in spatial domain
        plt.figure(2)
        plt.plot(Qs)
        plt.title('Q original in Spatial Domain')
        
        plt.show(block=False)
        
def display_yiq_freq(y, i, q):
        mag_y, mag_i, mag_q = yiq_frequency(y, i, q)
        
        
        # convert to dB for better results
        mag_y= 20 * np.log10(np.abs(mag_y))
        mag_i = 20 * np.log10(np.abs(mag_i))
        mag_q = 20 * np.log10(np.abs(mag_q))
        
        # Display the magnitude spectrum
        plt.figure(3)
        plt.semilogy(mag_y[::10])
        plt.title('Y original in Frequency Domain')
        
        # Display the magnitude spectrum
        plt.figure(4)
        plt.semilogy(mag_i[::10])
        plt.title('I original in Frequency Domain')
     
        
        # Display the magnitude spectrum
        plt.figure(5)
        plt.semilogy(mag_q[::10])
        plt.title('Q original in Frequency Domain')

        
        plt.show(block=False)
    
    
def yiq_spatial(y, i, q):
    
    # Reshape the channels to a 1D array
    Yspatial = y.flatten()
    Ispatial = i.flatten()
    Qspatial=  q.flatten()
    
    return Yspatial, Ispatial, Qspatial
    
def yiq_frequency(y, i, q):
    y, i, q = yiq_spatial(y, i, q)
    
    # Perform 1D FFT
    f_y = np.fft.fft(y)
    f_i = np.fft.fft(i)
    f_q = np.fft.fft(q)

    # Calculate the real parts
    y_frequency_domain = np.real(f_y)
    i_frequency_domain = np.real(f_i)
    q_frequency_domain = np.real(f_q)

    # Take the first half (due to symmetry)
    y_frequency_domain = y_frequency_domain[:len(y_frequency_domain)//2]
    i_frequency_domain = i_frequency_domain[:len(i_frequency_domain)//2]
    q_frequency_domain = q_frequency_domain[:len(q_frequency_domain)//2]

    
    return y_frequency_domain, i_frequency_domain, q_frequency_domain
    
def yiq(input):
    cap = get_video(input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(fps * 3)
    
    
    for i in range(num_frames):
        ret, frame = cap.read()

        # Check if the frame was read successfully.
        if not ret:
            break
        
        y, i, q = conv_to_yiq(frame)
        
        display_yiq(y, i, q)
        
        # Wait for a key press.
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        display_yiq_freq(y, i, q)
        
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
        
def display_rgb_freq(r, g, b):
    
        mag_r, mag_g, mag_b = rgb_frequency(r, g, b)
        
        
        # convert to dB for better results
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
        plt.title('B original in Frequency Domain')

        
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

    cap.release()
 
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

    # Take the first half
    R_frequency_domain = R_frequency_domain[:len(R_frequency_domain)//2]
    G_frequency_domain = G_frequency_domain[:len(G_frequency_domain)//2]
    B_frequency_domain = B_frequency_domain[:len(B_frequency_domain)//2]

    
    return R_frequency_domain, G_frequency_domain, B_frequency_domain


def process_yiq(y,i,q):
    Y = y.flatten()
    I = i.flatten()
    Q = q.flatten()

    # Perform FFT on Y, I, and Q
    f_y = np.fft.fft(Y, max_Frequency)
    f_i = np.fft.fft(I, max_Frequency)
    f_q = np.fft.fft(Q, max_Frequency)

    # Circular shift Y signal
    Ysignal = np.roll(f_y, Y_carrier_Frequency)
    

    # Zero out specified frequencies in Y, I, and Q
    Ysignal[I_beginning_Frequency:max_Frequency] = 0
    f_i[:Q_beginning_Frequency] = 0
    f_q[:Q_beginning_Frequency] = 0
    
    # Circular shift I and Q signals
    f_i = np.roll(f_i, I_carrier_Frequency)
    f_q = np.roll(f_q, Q_carrier_Frequency)



    # Combine I and Q signals
    IandQ = f_i + f_q

    # Add Y and I/Q
    compositeSignal = Ysignal + IandQ
    
    return compositeSignal
    
    
def composite_signal(input):
    cap = get_video(input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(fps * 3)
    
    for j in range(num_frames):
        ret, frame = cap.read()
        
        if not ret:
            break
        
        y, i, q = conv_to_yiq(frame)
        composite_image = process_yiq(y, i, q)
        composite_frames.append(composite_image)
        if j == 5:
            return

        
    
    #user = input("if you want to see the spatial or frequency domains of the composite signals, press c. To continue, press q")
    #if user == 'q':
    
        '''
        # Check if the frame was read successfully.
        if not ret:
            break
        
        y,i,q = conv_to_yiq(frame)
      
        composite_image = process_yiq(y, i, q)
    
        composite_spatial(composite_image)
        # Wait for a key press.
        cv2.waitKey(1)
        cv2.destroyAllWindows()
       
        
        composite_freq(composite_image)

        key = cv2.waitKey(1)
        if key == ord('0'):
            # Destroy all figures
            plt.close('all')
        elif key == ord('q'):
            break

    
    # Release the video capture object
    cap.release()
    # Destroy the windows.
    cv2.destroyAllWindows()
    
    '''
    
    
    
def composite_spatial(input):
    # Plot the result in spatial domain
    plt.figure(0)
    plt.plot(np.abs(input))
    plt.title('Composite in spatial domain')
        
    plt.show()
        
def composite_freq(input):
    
    #db = 20 * np.log10(np.abs(input))
    # Plot the result in freq domain
    plt.figure(0)
    plt.semilogy(np.real(input)[::10])
    plt.title('Composite in frequency domain')
        
    plt.show()
    
def display_recov_yiq(y, i, q):
    pass

def display_recov_yiq_freq(y, i, q):
   
    db_y = 20 * np.log10(np.real(y))
    db_i = 20 * np.log10(np.real(i))
    db_q = 20 * np.log10(np.real(q))
    
    plt.figure(0)
    plt.semilogy(db_y[::10])
    plt.title("Recovered Y in frequency domain")
    
    plt.figure(1)
    plt.semilogy(db_i[::10])
    plt.title("Recovered I in frequency domain")
    
    plt.figure(2)
    plt.semilogy(db_q[::10])
    plt.title("Recovered Q in frequency domain")
    
    plt.show()
    
    
    
def spatial_recov_yiq(y, i, q):
    y_recov = np.fft.ifft(y)
    i_recov = np.fft.ifft(i)
    q_recov = np.fft.ifft(q)
    
    plt.figure(0)
    plt.plot(y_recov)
    plt.title("Recovered Y in spatial domain")
    plt.ylim(0, 500)
    plt.xlim(0, 5000)
    
    plt.figure(1)
    plt.plot(i_recov)
    plt.title("Recovered I in spatial domain")
    
    plt.figure(2)
    plt.plot(q_recov)
    plt.title("Recovered Q in spatial domain")
    
    plt.show()
    
def recov_yiq_img(y, i, q):
   # Assuming 'freq_signal' is your 1D frequency domain signal

    # Step 1: Convert the 1D frequency domain signal to spatial domain
    y_image = np.fft.ifft(y).real

    y_image = np.reshape(y_image[:1080*1920], (1080, 1920))

    # Step 4: Display the image using cv2.imshow()
    cv2.imshow('Y Component Image', y_image)
    
    # Step 1: Convert the 1D frequency domain signal to spatial domain
    i_image = np.fft.ifft(i).real

    i_image = np.reshape(i_image[:1080*1920], (1080, 1920))

    # Step 4: Display the image using cv2.imshow()
    cv2.imshow('I Component Image', i_image)
    
     # Step 1: Convert the 1D frequency domain signal to spatial domain
    q_image = np.fft.ifft(q).real

    q_image = np.reshape(q_image[:1080*1920], (1080, 1920))

    # Step 4: Display the image using cv2.imshow()
    cv2.imshow('Q Component Image', q_image)

    

    
def recov_yiq():
    
    for comp in composite_frames:
        
        # for y recov
        y_recov = np.roll(np.pad(comp[1:I_beginning_Frequency-1], (0, max_Frequency - I_beginning_Frequency), 'constant', constant_values=0), -Y_carrier_Frequency)
        #i_recov = np.roll(np.roll(comp[I_beginning_Frequency:Q_beginning_Frequency-1],  -I_beginning_Frequency), -I_carrier_Frequency)
        i_recov = np.roll(np.pad(np.pad(comp[I_beginning_Frequency:Q_beginning_Frequency-1], (I_beginning_Frequency, 0), 'constant', constant_values=0), (0, max_Frequency-Q_beginning_Frequency), 'constant', constant_values=0), -I_carrier_Frequency)
        q_recov = np.roll(np.pad(comp[Q_beginning_Frequency:max_Frequency], (Q_beginning_Frequency-1, 0), 'constant', constant_values=0), -Q_carrier_Frequency)
        
        #y_recov = demodulate(comp, 'y')
        #i_recov = demodulate(comp, 'i')
        #q_recov = demodulate(comp, 'q')
        #display_recov_yiq_freq(y_recov, i_recov, q_recov)
        #spatial_recov_yiq(y_recov, i_recov, q_recov)
        recov_yiq_img(y_recov, i_recov, q_recov)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    
    






