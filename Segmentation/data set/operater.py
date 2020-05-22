import cv2
import math
import numpy as np
import os
import random

class Basic_Operator:

    #draw color contour 
    def draw_coordinates_color(img1,vx,vy,color):       
            if color ==0:
               painter  = [254,0,0]
            elif color ==1:
               painter  = [0,254,0]
            elif color ==2:
               painter  = [0,0,254]
            else :
               painter  = [0,0,0]
                        #path0  = signal.resample(path0, W)
            H,W,_ = img1.shape
            for j in range (len(vx)):
                    #path0l[path0x[j]]
                    dy = np.clip(vy[j],2,H-2)
                    dx = np.clip(vx[j],2,W-2)
                    img1[int(dy)+1,dx,:]=img1[int(dy)-1,dx,:]=img1[int(dy),dx,:]=painter
            return img1
