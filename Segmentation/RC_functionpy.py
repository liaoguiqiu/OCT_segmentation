
from analy import Save_signal_flag
import cv2
import math
import numpy as np
from median_filter_special import  myfilter
import pandas as pd
import os
 
import scipy.signal as signal
from scipy.stats.stats import pearsonr   
import random
from time import time
 
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
 
class RC_function(object):
     #formulars
     #n1* sin(that1)   = n2 sin (the) 
     def __init__(self):
         self.n  = 1.7  # refraction index between air and material 
         self.HV  = 20# the ratio between horizontal and vertical 
         self.pi = 3.14159
     def fill_up_contour(self,new,mask,img,contour):
        H,W = img.shape
        for i in range(W):
             new[0:int(contour[i])+50,i]   = img [0:int(contour[i])+50,i] 
             #new[0:500,i]   = img [0:500,i] 

             mask  [0:int(contour[i]),i]   = 0
        #cv2.imshow('correct',new.astype(np.uint8))
        #cv2.waitKey(1)
        return new,mask
     def angles(self,contour):
            L =len(contour)
            contour = -contour
            theta1 =  np.zeros((L))
            #theta1 = np.vectorize(theta1)
            tan1 = theta1
            tan1_0 = tan1
            tan1_1  =tan1
            tan1_2   = tan1
            for  i in range(10,L-11):
                # rememer to use the true distance  to change the factor
                # calculated with different scale 
                tan1_0[i] = (contour[i+2] - contour[i-2])/4/self.HV
                tan1_1[i] = (contour[i+5] - contour[i-5])/10/self.HV
                tan1_2[i] = (contour[i+10] - contour[i-10])/20/self.HV
                

            tan1  =  0.5*tan1_0  +  0.3*tan1_1  +  0.2*tan1_2
            theta1   = np.arctan(tan1)
            #  Snell's Law
            sin2= (1/ self.n)*np.sin(theta1)
            theta2  = np.arcsin(sin2)
            beta  = theta1- theta2
            #plt.plot(contour)
            #plt.plot(beta/self.pi *180, "--", color="gray")
            #plt.plot(theta1/self.pi *180, ":", color="gray")
            #plt.show()

            # beta the angle of the ray and vatical, and  this is a array vs each contour position 
            return beta
     def re_index(self, new,mask, img , contour , beta):
         H,W = img.shape
         L = len(contour)
         cos2  = np.cos(beta)
         sin2  = np.sin(beta)
         # ray tracing to put  pix to the correct ray position 
         for i in range( W):
             for j in range (int(contour[i]),H):
                 #if j > contour[i]: 
                     d = j- contour[i]
                     y_t  = d/self.n*cos2[i] + contour[i]
                     y_t = np.clip(y_t,0,H-1)
                     x_t=   d*sin2[i]  + i
                     x_t = np.clip(x_t,0,W-1)

                     new[int(y_t),int(x_t)] = img[j,i]
                     mask  [int(y_t),int(x_t)]  = 0



         return new,mask

    #  Snell's Law. for correction 
     def correct(self,img,contour):
        new= img # only calculate the front10 lines
        H,W = img.shape
        L = len(contour)
        contour   = signal.resample(contour , W)  

        new=new*0  # Nan pixel will be filled by intepolation processing
        mask =  new  + 255


        #1 copy all the pixeles up on the contour  to the new and make maks as 0
        new,mask   = self.fill_up_contour(new,mask,img,contour)
       
        #2 change every pix by line,  

        # beta the angle of the ray and vatical, and  this is a array vs each contour position 
        beta  = self. angles(contour)
        new,mask = self.re_index(new,mask,img,contour,beta)
        new=cv2.inpaint(new, mask, 2, cv2.INPAINT_TELEA)
        cv2.imshow('correct',new.astype(np.uint8))
        cv2.waitKey(1)
        return new