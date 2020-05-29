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
from scipy.ndimage import gaussian_filter1d
Manual_start_flag = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Seg_One_Frame(object):
     def __init__(self):
         
        self.operate_dir =   "../../saved_original_for_seg/503.jpg"
        self.savedir_path = "../../saved_processed/"
        self.display_flag = True
        from A_line import A_line_process
        self.aline = A_line_process()
        self.bias = 50
        self.aligh_flag =True
    #input image to calculate the fist aveline, for get the local minimal value calculation 
     def calculate_the_average_line(self,img):
        calculate_w = 5 # only calculate the front10 lines
        H,W = img.shape
        new = img[ :,5:5+calculate_w]
        ave_line=new.sum(axis=1)/calculate_w
        return ave_line
        #the  validation functionfor check the matrix and can also be used for validate the correction result
     def seg_process(self, Img ):
        gray  =   cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
        H,W   = gray.shape
        #gray = gray [: ,100: W -100]
        #Img = Img [: ,100: W -100]
        ave_line = self.calculate_the_average_line(gray)
        peaks  = self.aline.find_4peak(ave_line)

        H,W   = gray.shape


        #gray = cv2.blur(gray,(3,3))
        #gray = cv2.medianBlur(gray,5)
        #gray = cv2.blur(gray,(5,5))

        gray = cv2.GaussianBlur(gray,(3,3),0)
        #gray = cv2.bilateralFilter(gray,15,75,75)
        #gray = cv2.bilateralFilter(gray,15,75,75)
        ave_line = self.calculate_the_average_line(gray)
        peaks  = self.aline.find_4peak(ave_line)
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

        ave_line = self.calculate_the_average_line(sobel_y)
        peaks  = self.aline.find_4peak(ave_line) 

        Rever_img  = 255 - sobel_y
        
        
        #start_point = 596
        peaks = np.clip(peaks, 1,Rever_img.shape[0]-1)
        #new_peak = np.zeros(4)
        #new_peak=peaks
        #new_peak[3] = peaks[2]+35
        #peaks =  new_peak
        if Manual_start_flag == True:
            peaks[1] = peaks[0]+472-293
            peaks[2] = peaks[0]+500-293
            peaks[3] = peaks[0]+677-293


        path1,path_cost1=PATH.search_a_path(Rever_img,int(peaks[0]))
        path2,path_cost1=PATH.search_a_path(Rever_img,int(peaks[1]))
        path3,path_cost1=PATH.search_a_path(Rever_img,int(peaks[2]))
        path4,path_cost1=PATH.search_a_path(Rever_img,int(peaks[3]))
        path1 =gaussian_filter1d(path1,2)
        path2 =gaussian_filter1d(path2,2)

        path3 =gaussian_filter1d(path3,2)
        path4 =gaussian_filter1d(path4,2)

        path4=path3
        #path2 = path3
        path3 = path2+35

        [path1,path2,path3,path4]=np.clip([path1,path2,path3,path4],
                                          0,sobel_y.shape[0]-1)
        for i in range ( len(path1)):
             sobel_y[int(path1[i]),i]=254
             sobel_y[int(path2[i]),i]=254
             sobel_y[int(path3[i]),i]=254
             sobel_y[int(path4[i]),i]=254
        Dark_boundaries =  sobel_y *0
        [path1,path2,path3,path4]=np.clip([path1,path2,path3,path4],
                                          0,Dark_boundaries.shape[0]-2)
        for i in range ( len(path1)):
             Dark_boundaries[int(path1[i]),i]=254
             Dark_boundaries[int(path2[i]),i]=220
             Dark_boundaries[int(path3[i]),i]=199
             Dark_boundaries[int(path4[i]),i]=180


        [path1,path2,path3,path4]=np.clip([path1,path2,path3,path4],
                                          0,Img.shape[0]-2)
        for i in range ( Img.shape[1]):
             Img[int(path1[i])+1,i,:]=Img[int(path1[i]),i,:]=[254,0,0]
             Img[int(path2[i])+1,i,:]=Img[int(path2[i]),i,:]=[0,254,0]
             Img[int(path3[i])+1,i,:]=Img[int(path3[i]),i,:]=[0,0,254]
             Img[int(path4[i])+1,i,:]=Img[int(path4[i]),i,:]=[0,0,0]


        if self.display_flag == True:

            cv2.imshow('revert',Rever_img.astype(np.uint8))

            #cv2.imshow('path on blur',gray.astype(np.uint8))
            cv2.imshow('Seg2',Img)

            #sobel_y = sobel_y*0.1
            #edges = cv2.Canny(gray,50, 300,10)
            cv2.imshow('seg',sobel_y.astype(np.uint8))
            cv2.imwrite(self.savedir_path  + str(1) +".jpg",sobel_y .astype(np.uint8))


            cv2.waitKey(1) 
 
        return  Img,sobel_y,Dark_boundaries,[path1,path2,path3,path4]
        

if __name__ == '__main__':
    frame_process  = Seg_One_Frame()
    Img = cv2.imread(frame_process.operate_dir)  #read the first one to get the image size

    frame_process.seg_process(Img)
         