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
    
    r, g, b = cv2.split(input)

    # Normalize to the range [0, 1]
    #r = r / 255.0
    #g = g / 255.0
    #b = b / 255.0

    # Convert to float64 for precision
    #r = r.astype(np.float64)
    #g = g.astype(np.float64)
    #b = b.astype(np.float64)

    # RGB to YIQ conversion
    #y = 0.299*r + 0.587*g + 0.114*b
    #i = 0.596*r - 0.275*g - 0.321*b
    #q = 0.212*r - 0.523*g - 0.311*b
    
    yiq = cv2.cvtColor(input, cv2.COLOR_BGR2YCrCb)
    y, i, q = cv2.split(yiq)
        
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

        
        plt.show()
        
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
    Ysignal[I_beginning_Frequency:max_Frequency] = 0
    
    
     # Circular shift I and Q signals
    Isignal = np.roll(f_i, I_carrier_Frequency)


    # Zero out specified frequencies in Y, I, and Q
    
    Isignal[0:I_beginning_Frequency-1] = 0
    
    Qsignal= np.roll(f_q, Q_carrier_Frequency)
    Qsignal[0:Q_beginning_Frequency-1] = 0
    



    # Combine I and Q signals
    IandQ = Isignal + Qsignal

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
        if j == 2:
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
    Y_real = np.real(y[:1080*1920//2])
    I_real = np.real(i[:1080*1920//2])
    Q_real = np.real(q[:1080*1920//2])
    db_y = 20 * np.log10(Y_real)
    db_i = 20 * np.log10(I_real)
    db_q = 20 * np.log10(Q_real)
    
    plt.figure(0)
    plt.semilogy(db_y)
    plt.title("Recovered Y in frequency domain")
    
    plt.figure(1)
    plt.semilogy(db_i)
    plt.title("Recovered I in frequency domain")
    
    plt.figure(2)
    plt.semilogy(db_q)
    plt.title("Recovered Q in frequency domain")
    
    plt.show()
    
    
    
def spatial_recov_yiq(y, i, q):
    y_recov = np.fft.ifft(y)
    i_recov = np.fft.ifft(i)
    q_recov = np.fft.ifft(q)
    
    plt.figure(0)
    plt.plot(abs(y_recov))
    plt.title("Recovered Y in spatial domain")
    
    plt.figure(1)
    plt.plot(abs(i_recov))
    plt.title("Recovered I in spatial domain")
    
    plt.figure(2)
    plt.plot(abs(q_recov))
    plt.title("Recovered Q in spatial domain")
    
    plt.show()
    
    
    
def demodulate(comp):
    # Extract Y 
    YsignalRecovered = comp[1:I_beginning_Frequency-1]

    # Pad the array 
    YsignalRecovered = np.pad(YsignalRecovered, (0, max_Frequency-I_beginning_Frequency), mode='constant', constant_values=0)

    # Circular shift the signal
    YsignalRecovered = np.roll(YsignalRecovered, -Y_carrier_Frequency)

    i = comp[I_beginning_Frequency:Q_beginning_Frequency-1]

    # Pad the arrays
    i_recov = np.pad(i, (I_beginning_Frequency, 0), mode='constant', constant_values=0)
    i_recov_2 = np.pad(i_recov, (0, max_Frequency-Q_beginning_Frequency), mode='constant', constant_values=0)
    
    # Circular shift the signal
    IsignalRecovered = np.roll(i_recov_2, -I_carrier_Frequency)
    
     
    q = comp[Q_beginning_Frequency:max_Frequency]

    # Pad the array
    q = np.pad(q, (Q_beginning_Frequency-1, 0), mode='constant', constant_values=0)

    # Circular shift the signal
    qQsignalRecovered = np.roll(q, -Q_carrier_Frequency)
    
    return YsignalRecovered, IsignalRecovered, qQsignalRecovered

    
def recov_yiq_img(y, i, q):
    
    y_image = np.reshape(y[:1080*1920], (1080, 1920))
    plt.figure(0)
    plt.imshow(np.abs(y_image), cmap='gray')
    y_image = np.abs(y_image)
    
    
    plt.figure(1)
    i_image = np.reshape(i[:1080*1920], (1080, 1920))
    plt.imshow(np.abs(i_image), cmap='gray')
    i_image = np.abs(i_image)
    
    plt.figure(2)
    q_image = np.reshape(q[:1080*1920], (1080, 1920))
    plt.imshow(np.abs(q_image), cmap='gray')
    q_image = np.abs(q_image)

    
    plt.show()
    

    #return y_image, i_image, q_image
    

    
def recov_yiq():
    
    for comp in composite_frames:
        y_recov, i_recov, q_recov = demodulate(comp)
        #display_recov_yiq_freq(y_recov, i_recov, q_recov)
        #spatial_recov_yiq(y_recov, i_recov, q_recov)
        y_1D = np.fft.ifft(y_recov)
        i_1D = np.fft.ifft(i_recov)
        q_1D = np.fft.ifft(q_recov)
        #recov_yiq_img(y_1D, i_1D, q_1D)
        y_image = np.abs(np.reshape(y_1D[:1080*1920], (1080, 1920)))
        i_image = np.abs(np.reshape(i_1D[:1080*1920], (1080, 1920)))
        q_image = np.abs(np.reshape(q_1D[:1080*1920], (1080, 1920)))
        r,g,b = recov_yiq_rgb(y_image, i_image, q_image)
        #recov_rgb_spatial(r,g,b)
        display_rgb_freq(r, g, b)
        
        # Wait for a key press.
        key = cv2.waitKey(0)
        
        if key == ord('0'):
            # Destroy all figures
            plt.close('all')
        elif key == ord('q'):
            break
    
    
    
    
def recov_yiq_rgb(y, i, q):
    
    r = 1.0*y + 0.956*i + 0.620*q
    g = 1.0*y-0.272*i-0.647*q
    b = 1.0*y-1.108*i+ 1.700*q


    
    return r, g, b

def recov_rgb_spatial(r, g, b):
     # We will now represent each frame in R G B image form
        # Display the R, G, and B frames.
        
        plt.figure(0)
        plt.imshow(r, cmap='gray')
        
        plt.figure(1)
        plt.imshow(g, cmap='gray')
        
        plt.figure(2)
        plt.imshow(b, cmap='gray')
        
      
        #image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        #image[:, :, 0] = (r)
        #image[:, :, 1] = (g)
        #image[:, :, 2] = (b)
        
       #plt.figure(3)
        #plt.imshow(image)
#       
        
        #plt.show()
        #cv2.imshow("image", rgb)  
        #return
    
        Rs, Gs, Bs = rgb_spatial(r, g, b)
        
        # Plot the result in spatial domain
        plt.figure(4)
        plt.plot(Rs)
        plt.title('R original in Spatial Domain')

        # Plot the result in spatial domain
        plt.figure(5)
        plt.plot(Gs)
        plt.title('G original in Spatial Domain')
        
        # Plot the result in spatial domain
        plt.figure(6)
        plt.plot(Bs)
        plt.title('B original in Spatial Domain')
        
        plt.show()
        