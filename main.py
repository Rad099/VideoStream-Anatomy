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

    
    # Original RGB components
    #util.rgb(video)

    # Orgiinal YIQ components
    #util.yiq(video)


    # Composite signal
    util.composite_signal(video)
    
    
    # Recovered YIQ


    # Recovered RGB


if __name__ == "__main__":
    main()
    
