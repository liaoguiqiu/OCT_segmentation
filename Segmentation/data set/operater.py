import cv2
import math
import numpy as np
import os
import random
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d

class Basic_Operator:

        # use the H and W of origina to confine , and generate a random reseanable signal in the window
    def random_shape_contour(H,W,x,y):
        # first need to determine whether use the origina lcountour to shift 
        dc1 =np.random.random_sample()*10
        dc1  = int(dc1)%2
        dc1=0
        if dc1==0: # use the original signal 
            # inital ramdon width and height
            width =  int((0.3+0.5* np.random.random_sample())*W)
            height =  int((0.3+0.5* np.random.random_sample())*H)
            # star and end
            dx1 = int(  np.random.random_sample()*(W-width)  )
            dy1 = int(  np.random.random_sample()*(H-height) )
            dx2  = dx1+width
            dy2  = dy1+height

            #new x
            newx = np.arange(dx1, dx2)

            #new y based on a given original y
            newy=signal.resample(y, width)
            r_vector   = np.random.sample(20)*20
            r_vector=signal.resample(r_vector, width)
            r_vector = gaussian_filter1d (r_vector ,10)
            newy = newy + r_vector
            miny=min(newy)
            height0  = max(newy)-miny
            newy = (newy-miny) *height/height0 + dy1 

            pass
        else:
            pass
        Dice = int( np.random.random_sample()*10)
        if Dice % 2 ==0 :
            newy = y
            newx = x
        #width  = 30% - % 50


        #sample = np.arange(width)
        #r_vector   = np.random.sample(20)*20
        #r_vector = gaussian_filter1d (r_vector ,10)
        #newy = np.sin( 1*np.pi/width * sample)
        #newy = -new_contoury*(dy2-dy1)+dy2
        #newy=new_contoury+r_vector
        #newx = np.arange(dx1, dx2)
        return newx,newy
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
    def gray2rgb(img):
        new=np.zeros((img.shape[0],img.shape[1],3))
        new[:,:,0]  = img
        new[:,:,1]  = img
        new[:,:,2]  = img

        return new
    def warp_padding(img,contour0,new_contour):
        shift_vector =  new_contour  - contour0
        new_image  = img
        for iter in range(len(shift_vector)):
            lineshift= int(shift_vector[iter] )
            new_image[:,iter] =np.roll( img[:,iter] ,lineshift)
            # The carachter within the contour need special stratigies to maintain 
            if(lineshift>0):#
                origin_point  = int(contour0[iter])
                new_image[0:lineshift,iter]= signal.resample(img[0:origin_point,iter], 
                                                             lineshift)
                pass
        return new_image
        #just roll one line 
    def warp_padding_line1(line,y0,new_y):
        shift  =  int(new_y  - y0)
        new_y = int(new_y)
        y0 = int (y0)
        line_new =np.roll( line ,shift)
        # The carachter within the contour need special stratigies to maintain 
        if(shift>0):#
            line_new[0:new_y]= signal.resample(line[0:y0],new_y)           
        return line_new
    def warp_padding_line2(line,y0,new_y):
        shift  =  int(new_y  - y0)
        new_y = int(new_y)
        y0 = int (y0)
        line_new =np.roll( line ,shift)
        # The carachter within the contour need special stratigies to maintain 
        if(shift>0):#
            line_new[0:new_y]= signal.resample(line[int(y0/2):y0],new_y)           
        return line_new

    # a paathech
    def generate_patch_with_contour(img1,H_new,contour0x,contour0y,
                                    new_contourx,new_contoury):
        H,W  = img1.shape
        img1 = cv2.resize(img1, (W,H_new), interpolation=cv2.INTER_AREA)
        contour0y = contour0y*H_new/H
        points = len(contour0x)
        points_new = len(new_contoury)
        W_new  = points_new
        new  = np.zeros((H_new,W_new))
        for i in range(W_new):
            line_it = int( np.random.random_sample()*points)
            line_it = np.clip(line_it,0,points-1) 
            source_line = img1[:,contour0x[line_it]]
            #new[:,i] = ba.warp_padding_line1(source_line, contour0y[line_it],new_contoury[i])
            new[:,i] = Basic_Operator .warp_padding_line2(source_line, contour0y[line_it],new_contoury[i])
            #random select a source
            pass
        pass
        return new
    def generate_patch_base_origin(img1,H_new,contour0x,contour0y,
                                    new_contourx,new_contoury):
        H,W  = img1.shape


        contour0y = contour0y*H_new/H
        points = len(contour0x)
        points_new = len(new_contoury)
        W_new  = points_new
        original_patch  =  img1[:,contour0x[0]:contour0x[points-1]]
        original_patch  =  cv2.resize(original_patch, (points_new,H_new), interpolation=cv2.INTER_AREA)
        contour0y=signal.resample(contour0y, W_new)
        img1 = cv2.resize(img1, (W,H_new), interpolation=cv2.INTER_AREA)
        

        new  = np.zeros((H_new,W_new))
        for i in range(W_new):
            #line_it = int( np.random.random_sample()*points)
            #line_it = np.clip(line_it,0,points-1) 
            source_line = original_patch[:,i]
            #new[:,i] = ba.warp_padding_line1(source_line, contour0y[line_it],new_contoury[i])
            new[:,i] = Basic_Operator .warp_padding_line2(source_line, contour0y[i],new_contoury[i])
            #random select a source
            pass
        pass
        return new
        #fill in a long vector with a small er on e
    #this bversion just use the resample
    #next version can use the clone
    def fill_lv_with_sv1(sv,H):
        #h = len(sv)
        #div = (H/h)
        #if div > 2:
        #    for i in range(3):
        #        sv=np.append(sv,sv)
        lv=signal.resample(sv, H)     
        return lv
    def fill_lv_with_sv2(sv,H):
        lv = np.zeros(H)
        h = len(sv)
        div = (H/h)
        div= int(math.log2(div))+1
        for i in range(div):
            sv=np.append(sv,sv)
        lv = sv[0:H] 
        return lv
    # to generate synthetic background with this image with label 
    def generate_background_image2(img,contourx,contoury,H,W):
        
        points = len(contourx)
        new  = np.zeros((H,W))
        #generate line by line 
        for i in range(W):
            #random select a source
 
            #random select a A-line
            line_it = int( np.random.random_sample()*points)
            line_it = np.clip(line_it,0,points-1) 
            y = contoury[line_it]
            #pick part of the A-line betwenn contour and scanning center
            source_line = img[int(0.3*y):int(0.6*y),contourx[line_it]]

            #source_h  = h_list[pic_it]
            #new[:,i] = self.fill_lv_with_sv1(source_line,H)
            new[:,i] =  Basic_Operator .fill_lv_with_sv1(source_line,H)

        return new