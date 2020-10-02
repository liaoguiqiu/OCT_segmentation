# 2020 5:00 2 nd oct
# this is a subfunction  of seg imag
# 
# 

from analy import Save_signal_flag
import cv2
import math
import numpy as np
from median_filter_special import  myfilter
import pandas as pd
import os
import torch
import scipy.signal as signal
from scipy.stats.stats import pearsonr   
import random
from time import time
from  path_finding import PATH
from scipy.ndimage import gaussian_filter1d
Manual_start_flag = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Seg_One_1(object):
     def __init__(self):
         
        self.operate_dir =   "../../saved_original_for_seg/503.jpg"
        self.savedir_path = "../../saved_processed/"
        self.display_flag = True
        from A_line import A_line_process
        self.aline = A_line_process()
        self.bias = 5
        self.region = 50 # the region in vertial
        self.aligh_flag =True
     def calculate_the_average_line(self,img,start_p):
        calculate_w = 5 # only calculate the front10 lines
        H,W = img.shape
        new = img[ start_p-self.region:start_p+self.region,5:5+calculate_w]
        ave_line=new.sum(axis=1)/calculate_w
        return ave_line
        #the  validation functionfor check the matrix and can also be used for validate the correction result
     def seg_process(self, Img,start_p ):
        gray  =   cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
        H,W   = gray.shape
        #gray = gray [: ,100: W -100]
        #Img = Img [: ,100: W -100]
         

        H,W   = gray.shape


        #gray = cv2.blur(gray,(3,3))
        #gray = cv2.medianBlur(gray,5)
        #gray = cv2.blur(gray,(5,5))

        gray = cv2.GaussianBlur(gray,(3,3),0)
        #gray = cv2.bilateralFilter(gray,15,75,75)
        #gray = cv2.bilateralFilter(gray,15,75,75)
        ave_line = self.calculate_the_average_line(gray,start_p)
        #peaks  = self.aline.find_4peak(ave_line)
        #gray = cv2.blur(gray,(5,5),0)
      
        if self.display_flag == True :
            cv2.imshow('ini',Img)


            cv2.imshow('blur',gray.astype(np.uint8))


        x_kernel = np.asarray([[-1, 0, 1], # Sobel kernel for x-direction
                      [-2, 0, 2],
                      [-1, 0, 1]])

        #y_kernel = np.asarray([[-2, -2, -2], # Sobel kernel for y-direction
        #                       [1,   1,  1],
        #                       [1,   1,  1]])
        y_kernel = np.asarray([[-1,-1,-1], # Sobel kernel for y-direction
                               [-1,-1,-1],
                               [-1,-1,-1],
                               [6,6,6],
                               [-1,-1,-1],
                               [-1,-1,-1],
                               [ -1,-1,-1]])
        #y_kernel = np.asarray([ # Sobel kernel for y-direction
        #                [-1,-1],

        #                [-1,-1],
        #                [-1,-1],
        #                [-1,-1],

        #                [12,12],
        #                [-1,-1],
        #                [-1,-1],
        #                [-1,-1],
        #                [-1,-1],
        #                [-1,-1],
        #                [-1,-1],
        #                [-1,-1],
        #                [-1,-1],


                         
        #                ])
        y_kernel = np.asarray([ # Sobel kernel for y-direction
                        [-1,0 ],

                        [-1,0 ],
                        [-1,0 ],
                        [-1,0 ],

                        [28,0 ],
                        [-1,0 ],
                        [-1,0 ],
                        [-1 ,0],
                        [-1,0 ],
                        [-1,0 ],
                        [-1,0 ],
                        [-1 ,0],
                        [-1,0 ],
                        [-1,0 ], 
                        [-1,0 ],   
                        [-1,0 ],   
                        [-1,0 ],   
                         [-1 ,0],
                        [-1,0 ],
                        [-1,0 ], 
                        [-1,0 ],   
                        [-1,0 ],   
                        [-1,0 ],
                        [-1 ,0],
                        [-1,0 ],
                        [-1,0 ], 
                        [-1,0 ],   
                        [-1,0 ],   
                        [-1,0 ],

                        ])
        y_kernel = y_kernel/9
        gray = gray.astype(np.float)              
        #sobel_x = signal.convolve2d(gray, x_kernel) #
        sobel_y = signal.convolve2d(gray, y_kernel) # convolve kernels over images
        sobel_y = np.clip(sobel_y+10, 1,254)
        sobel_y = cv2.medianBlur(sobel_y.astype(np.uint8),5)
        sobel_y = cv2.GaussianBlur(sobel_y,(5,5),0)
        sobel_y = cv2.blur(sobel_y,(5,5))

        #sobel_y = cv2.GaussianBlur(sobel_y,(5,5),0)
        #sobel_y = cv2.bilateralFilter(sobel_y.astype(np.uint8),9,175,175)
        #find the start 4 point

        sobel_y=sobel_y[self.bias:H-self.bias, :]
        Img=Img[self.bias:H-self.bias, :]
        Rever_img  = 255 - sobel_y

        #ave_line = self.calculate_the_average_line(sobel_y)
        #peaks  = self.aline.find_4peak(ave_line) 

        ave_line = self.calculate_the_average_line(gray,start_p)
        peaks  = self.aline.find_4peak(ave_line)
        
        #start_point = 596
        peaks = np.clip(peaks, 1,Rever_img.shape[0]-1)
        #new_peak = np.zeros(4)
        #new_peak=peaks
        #new_peak[3] = peaks[2]+35
        #peaks =  new_peak
         


        path1,path_cost1=PATH.search_a_path(Rever_img,start_p- self.region + peaks[0])
         
        path1 =gaussian_filter1d(path1,2)
        path1  =  np.clip(path1,  0,sobel_y.shape[0]-1)
         
        for i in range ( len(path1)):
             sobel_y[int(path1[i]),i]=254
        
         
 

        if self.display_flag == True:

            cv2.imshow('revert',Rever_img.astype(np.uint8))

            #cv2.imshow('path on blur',gray.astype(np.uint8))
            cv2.imshow('Seg2',Img)

            #sobel_y = sobel_y*0.1
            #edges = cv2.Canny(gray,50, 300,10)
            cv2.imshow('seg',sobel_y.astype(np.uint8))


            cv2.waitKey(1) 
 
        return  path1
        

if __name__ == '__main__':
    frame_process  = Seg_One_Frame()
    Img = cv2.imread(frame_process.operate_dir)  #read the first one to get the image size

    frame_process.seg_process(Img)
