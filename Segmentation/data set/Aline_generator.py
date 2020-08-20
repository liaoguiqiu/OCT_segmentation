import cv2
import math
import numpy as np
import os
import random
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
class A_line_syn():
        
    def __init__(self):
        self.plot_flag   = True
        self.pixnum = 1024
        #boundary position
        self.bound_pos=[300,600,900] # the first layer is air
        self.draw_fig_flag = False # the flag for analy drawing
        self.y=[]
        # import the draw signal package
    def generator(self):
        x = np.arange(0, self.pixnum)
        y1 = np.exp(-x/300)
        y2  = 2*np.exp(-x/300)
        y3 = 3*np.exp(-x/300)
        window1 = np.append(np.zeros(self.bound_pos[0]),np.ones(self.bound_pos[1]-self.bound_pos[0]))
        window1 = np.append(window1,np.zeros(self.pixnum-self.bound_pos[1]))
        window2 = np.append(np.zeros(self.bound_pos[1]),np.ones(self.bound_pos[2]-self.bound_pos[1]))
        window2 = np.append(window2,np.zeros(self.pixnum-self.bound_pos[2]))
        window3 = np.append(np.zeros(self.bound_pos[2]),np.ones(self.pixnum-self.bound_pos[2]))

        y  =  np.multiply ( y1 , window1) + np.multiply ( y2 , window2) + np.multiply ( y3 , window3)

        self.y  = y 
        return self.y
    def plot(self):
        if self.plot_flag == True:
     
            plt.plot(self.y)
            pass
        pass
if __name__ == '__main__':
    generator = A_line_syn()
    y  = generator.generator()
    generator.plot()


         