operatedir_one  = "../../saved_original_for_seg/496.jpg"

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Seg_One_Frame(object):
     def __init__(self):
         
        self.operate_dir =   "../../saved_original_for_seg/496.jpg"
        self.savedir_path = "../../saved_processed/"


        #the  validation functionfor check the matrix and can also be used for validate the correction result
     def seg_process(self ):
        Img = cv2.imread(self.operate_dir)  #read the first one to get the image size
        gray  =   cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('ini',gray.astype(np.uint8))
        H,W   = gray.shape
        gray = gray [: ,100: W -100]
        #gray = cv2.blur(gray,(3,3))
        #gray = cv2.blur(gray,(3,3))
        gray = cv2.medianBlur(gray,3)
        gray = cv2.GaussianBlur(gray,(5,5),0)
        gray = cv2.bilateralFilter(gray,9,75,75)
        #gray = cv2.blur(gray,(5,5),0)
      



        cv2.imshow('blur',gray.astype(np.uint8))


        x_kernel = np.asarray([[-1, 0, 1], # Sobel kernel for x-direction
                      [-2, 0, 2],
                      [-1, 0, 1]])

        #y_kernel = np.asarray([[-2, -2, -2], # Sobel kernel for y-direction
        #                       [1,   1,  1],
        #                       [1,   1,  1]])
        y_kernel = np.asarray([[-1], # Sobel kernel for y-direction
                               [-1],
                               [-1],
                               [6],
                               [-1],
                               [-1],
                               [ -1]])

        gray = gray.astype(np.float)              
        #sobel_x = signal.convolve2d(gray, x_kernel) #
        sobel_y = signal.convolve2d(gray, y_kernel) # convolve kernels over images
        sobel_y = np.clip(sobel_y, 1,254)
        sobel_y = cv2.GaussianBlur(sobel_y,(5,5),0)
        Rever_img  = 255 - sobel_y
        #start_point = 190
        start_point = 596

        path1,path_cost1=PATH.search_a_path(Rever_img,start_point)
        for i in range ( len(path1)):
             sobel_y[int(path1[i]),i]=254
        cv2.imshow('revert',Rever_img.astype(np.uint8))

        #sobel_y = sobel_y*0.1
        #edges = cv2.Canny(gray,50, 300,10)
        cv2.imshow('seg',sobel_y.astype(np.uint8))
        cv2.imwrite(self.savedir_path  + str(1) +".jpg",sobel_y .astype(np.uint8))

        cv2.waitKey(1) 
 
        return  Img
        

if __name__ == '__main__':
    frame_process  = Seg_One_Frame()
    frame_process.seg_process()
         