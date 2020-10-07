import cv2
import math
import numpy as np
import os
import random
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from operater import Basic_Operator


class Basic_Operator2:
    # use the H and W of origina to confine , and generate a random reseanable signal in the window
    
    def random_sheath_contour(H,W,x,y):
        # first need to determine whether use the origina lcountour to shift

        # random rol the sheath 
         
        np.roll(y, int(np.random.random_sample()*len(y)-1)) 

        dc1 = np.random.random_sample()*10
        dc1  = int(dc1)%2
        if dc1==0: # not use the original signal 
            # inital ramdon width and height
           
            # should mainly based o the sheath orginal contor
            newy = signal.resample(y, W)
            newx = np.arange(0, W)
            r_vector   = np.random.sample(20)*10
            r_vector=signal.resample(r_vector, W)
            r_vector = gaussian_filter1d (r_vector ,10)
            newy = newy + r_vector 
                    
            newy = np.clip(newy,50,H-1)
        else:       
            newy = signal.resample(y, W)
            newx = np.arange(0, W)
        #width  = 30% - % 50


        #sample = np.arange(width)
        #r_vector   = np.random.sample(20)*20
        #r_vector = gaussian_filter1d (r_vector ,10)
        #newy = np.sin( 1*np.pi/width * sample)
        #newy = -new_contoury*(dy2-dy1)+dy2
        #newy=new_contoury+r_vector
        #newx = np.arange(dx1, dx2)
        return newx,newy
    def random_shape_contour(H,W,sx,sy,x,y):
        # determine the tissue contour based o hte determined sheath contour
        dc1 =np.random.random_sample()*10
        dc1  = int(dc1)%2
        if dc1==0: # use the original signal 
            # inital ramdon width and height
           
            width =  int((0.05+0.91* np.random.random_sample())*W)
            dx1 = int(  np.random.random_sample()*(W-width)  )
            dx2  = dx1+width
            dy1 = int(  np.random.random_sample()*H*1.5 -0.25*H)
            dy2  = int  ( np.random.random_sample()*(H*1.5-dy1)) + dy1

            height =  dy2-dy1
            # star and end
            #new x
            newx = np.arange(dx1, dx2)
            #new y based on a given original y
            newy=signal.resample(y, width)
            r_vector   = np.random.sample(20)*50
            r_vector=signal.resample(r_vector, width)
            r_vector = gaussian_filter1d (r_vector ,10)
            newy = newy + r_vector
            miny=min(newy)
            height0  = max(newy)-miny
            newy = (newy-miny) *height/height0 + dy1 
        else:       
            newy = y
            newx = x
        #limit by the bondary of the sheath
        for i in range(len( newy ) ):

            newy[i]  = np.clip(newy[i] , sy[newx[i]]-1,H-1) # allow it to merge int o 1 pix
        #width  = 30% - % 50
        newy = np.clip(newy,0,H-1)
        
        return newx,newy
    #draw color contour 
    def fill_sheath_with_contour(img1,H_new,W_new,contour0x,contour0y,
                                    new_contourx,new_contoury):
        H,W  = img1.shape
        img1 = cv2.resize(img1, (W_new,H_new), interpolation=cv2.INTER_AREA)
        contour0y = contour0y*H_new/H
        points = len(contour0x)
        points_new = len(new_contoury)
        #W_new  = points_new
        new  = np.zeros((H_new,W_new))
        mask =  new  + 255
        contour0y=signal.resample(contour0y, W_new)
        contour0x=np.arange(0, W_new)
        # use a dice to determin wheterh follw orgina sequnce 
        Dice = int( np.random.random_sample()*10)

        for i in range(W_new):
            

            if Dice % 3 ==0 : # less possibility to random select the A line s 
                line_it = int( np.random.random_sample()*points)
                line_it = np.clip(line_it,0,points-1) 
                
            else: 
                line_it   =  i
            #line_it   =  i


            source_line = img1[:,contour0x[line_it]]
            #directly_fillinnew
            newy   = int(new_contoury[i] )
            iniy   =  int (contour0y[line_it]) + 5   # add 5 to give more high light bondaries 
            shift  =  int(newy - iniy)
            if shift < 0:
                new[0:newy,i] = source_line[-shift:iniy]
                mask[0:newy,i] = 0
            else :
                new[shift:newy,i] = source_line[0:iniy]
                mask[shift:newy,i]  = 0

             
            pass
        pass
        return new,mask
    def fill_patch_base_origin(img1,H_new,contour0x,contour0y,
                                    new_contourx,new_contoury,new,mask):
        H,W  = img1.shape
        contour0y = contour0y*H_new/H
        points = len(contour0x)
        points_new = len(new_contoury)
        W_new  = points_new
        # resize the patch has countour to the target contour size
        original_patch  =  img1[:,contour0x[0]:contour0x[points-1]]
        original_patch  =  cv2.resize(original_patch, (points_new,H_new), interpolation=cv2.INTER_AREA)
        contour0y=signal.resample(contour0y, points_new)
        img1 = cv2.resize(img1, (W,H_new), interpolation=cv2.INTER_AREA)
        
        #new  = np.zeros((H_new,W_new))
        for i in range(points_new):
            #line_it = int( np.random.random_sample()*points)
            #line_it = np.clip(line_it,0,points-1) 
            line_it = i
            source_line = original_patch[:,line_it]
            #new[:,i] = ba.warp_padding_line1(source_line, contour0y[line_it],new_contoury[i])
            #new[:,i] = Basic_Operator .warp_padding_line2(source_line, contour0y[i],new_contoury[i])
            #random select a source
            newy   = int(new_contoury[i] )
            iniy   =  int (contour0y[line_it]) - 5   # add 5 to give more high light bondaries 
            shift  =  int(newy - iniy)
            if shift > 0:
                new[newy:H_new,new_contourx[i]] = source_line[iniy:H_new-shift]
                mask[newy:H_new,new_contourx[i]] = 0
            else :
                new[newy:H_new+shift,new_contourx[i]] = source_line[iniy:H_new ]
                mask[newy:H_new+shift,new_contourx[i]]  = 0

 
        return new,mask
             
     