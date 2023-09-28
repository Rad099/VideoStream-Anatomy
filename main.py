'''
Ridwan Alrefai - Matteo
University of Illinois Chicago
main.py
   
   
The main file has the main 4 steps of the project, and at each step, presents the spatial and frequency data
for the component specified.
   
   
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
import util
import time


video = "IMG_4148.mp4"

def main():
    print(f'Welcome! We will be going through each of the steps of this computer project.')
    # THis project has become more messy than expected. To get the info for each part, uncomment the compoenent name you wish to see. For 
    # the recovered components, you need the util.composite_signal(video) uncommented along with util.recov_yiq for both recovered yiq and rgb.
    # For this, uncomment whatever you want to see in util.py (i.e freq, spatial etc.). The util functions are named comprehensviley 
    # for easy understanding
    
    # Original RGB components
    util.rgb(video)

    # Orgiinal YIQ components
    util.yiq(video)


    # Composite signal
    util.composite_signal(video)
    
    
    # Recovered YIQ and Recovered RGB
    util.recov_yiq()
    





if __name__ == "__main__":
    main()
    
