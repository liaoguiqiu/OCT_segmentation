import cv2
import math
import numpy as np
import os
import random
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from operater import Basic_Operator
from scipy import fftpack

class Basic_Operator2:

    def add_original_back (new,back,mask):
        mask = np.clip(mask,0,1)
        new = new + back*mask
        # create mask1
        #keep thebakcground part 
 
        return new
    def pure_background (img,contourx,contoury,H,W):
        # use different strategies:
        ori_H,ori_W  = img.shape
        points = len(contourx[1])
        new  = np.zeros((H,W))
        c_len = len(contourx[1]) # use the send 
        #c_len0 = len(contourx[0]) # use the send 

        #Dice = int( np.random.random_sample()*10)
        #method 1 just use the left and right side of the imag to raasampel
        

        min_b  = int(np.max(contoury[0]))
        max_b  = int(np.min(contoury[1]))
        if  c_len<0.8* ori_W:

            sourimg1  = img[:,0:contourx[1][0]]
            sourimg2  = img[:,contourx[1][c_len-2]: ori_W]
            sourimg = np.append(sourimg2,sourimg1,axis =1) # the right sequence 
            sr_H,sr_W  = sourimg.shape
            pend_cnt  = int(W/sr_W)+1
            pender  =   cv2.flip(sourimg, 1)
            new  = sourimg
            for i in range(pend_cnt):
                new  = np.append(new,pender, axis=1) # cascade
                pender  =   cv2.flip(pender, 1)
            min_b  = int(np.max(contoury[0]))   
            out  = new[min_b:ori_H,0:W] # crop out the sheth
            out  = cv2.resize(out, (W,H), interpolation=cv2.INTER_LINEAR )
        else :
            if (max_b -min_b)>200 :
                #method 2 the line is generated with the line above the the contour 
                #generate line by line 
                #min_b  = int(np.max(contoury[0]))
                #max_b  = int(np.min(contoury[1]))
                #if (max_b -min_b)>200:
                sourimg  = img[min_b:max_b,:]
                sr_H,sr_W  = sourimg.shape
                pend_cnt  = int(H/sr_H)+1
                pender  =   cv2.flip(sourimg, 0)
                new  = sourimg
                for i in range(pend_cnt):
                    new  = np.append(new,pender, axis=0) # cascade
                    pender  =   cv2.flip(pender, 0)

                out  = new 
                out  = cv2.resize(out, (W,H), interpolation=cv2.INTER_LINEAR )
            else:  # del with this special condition when full sorround contour 
                #left_a = np.max([contourx[0][0],contourx[1][0]])
                #right_a = np.max([contourx[0][0],contourx[c_len][0]])
                index = 0
                source_i  = 0
                sourimg   = np.zeros((ori_H,50))
                #calculate the with between 2 bondaries
                py1_py2 = contoury[1]  - contoury[0]
                max_d  = int(0.8*np.max(py1_py2)) 
                sourimg   = np.zeros((max_d,50)) #  create a block based on the area

                while(1):
                    if ( contoury[1][index] - contoury[0][index] )> max_d+5:
                        sourimg[:,source_i]  = img[int(contoury[0][index]):int(contoury[0][index])+max_d, contourx[1][index]]
                        source_i+=1
                        if  source_i >= 50:
                            break
                    index +=1 
                    if index >= len(contoury[1]):
                        index=0


                sr_H,sr_W  = sourimg.shape
                pend_cnt  = int(W/sr_W)+1 # pend through horizontal
                pender  =   cv2.flip(sourimg, 1)
                new  = sourimg
                for i in range(pend_cnt):
                    new  = np.append(new,pender, axis=1) # cascade
                    pender  =   cv2.flip(pender, 1)
                #min_b  = int(np.max(contoury[0]))   
                out  = new[:,0:W] # crop out the sheth
                out  = cv2.resize(out, (W,H), interpolation=cv2.INTER_LINEAR )
        return out

    # use the H and W of origina to confine , and generate a random reseanable signal in the window
    def upsample_background (img,H_new,W_new):
        # use fft to upsample 
        H,W  = img.shape
        im_fft = fftpack.fft2(img)
        im_fft2 = im_fft.copy()
        H,W  = img.shape

        LR =  np.zeros((H,int((W_new - W)/2)))
        new  = np.append(LR,im_fft2,axis=1) # cascade
        new  = np.append(new,LR,axis=1) # cascade
        H,W  = new.shape
        TB = np.zeros(((int((H_new - H)/2)), W))
        new  = np.append(TB,new,axis=0) # cascade
        new  = np.append(new,TB,axis=0) # cascade
        new_img  = fftpack.ifft2(im_fft2).real
        new_img = cv2.resize(img, (W_new,H_new), interpolation=cv2.INTER_AREA)

        return new_img

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
    def random_shape_contour(H_ini,W_ini,H,W,sx,sy,x,y):
        # determine the tissue contour based o hte determined sheath contour
        dc1 =np.random.random_sample()*100
        dc1  = int(dc1)%10
        if dc1!=0: # use the original signal 
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
        if len(x) > 0.96 *W_ini: # consider the special condition of full and gapin middel 

            width =  W
            dx1 = 0
            dx2  = W
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


            # rememver to add resacle later
            #newy = signal.resample(y, W)
            #newx = np.arange(0, W)
            ##np.roll(y, int(np.random.random_sample()*len(y)-1)) 
            #newy  = newy +  np.random.random_sample() *H/2

            # and also deal with black area
            for i in range(len( y ) ):
                if y[i] >= (H_ini-5):
                    newy[i]  = H-1 # allow it to merge int o 1 pix

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
            iniy   =  int (contour0y[line_it]) - 3   # add 5 to give more high light bondaries 
            shift  =  int(newy - iniy)
            if shift > 0:
                new[newy:H_new,new_contourx[i]] = source_line[iniy:H_new-shift]
                mask[newy:H_new,new_contourx[i]] = 0
            else :
                new[newy:H_new+shift,new_contourx[i]] = source_line[iniy:H_new ]
                mask[newy:H_new+shift,new_contourx[i]]  = 0

 
        return new,mask
    # deal with non full connected path, transfer these blank area         
    def re_fresh_path( px,py,H,W):
        # this function input the original coordinates of contour x and y, orginal image size and out put size

        if len(px) > 0.96 *W: # first consider the special condition of full and gapin middel 
            # rememver to add resacle later
            new_y = signal.resample(py, W)
            new_x = np.arange(0, W)
            return new_x,new_y

            ##np.roll(y, int(np.random.random_sample()*len(y)-1)) 
            #newy  = newy +  np.random.random_sample() *H/2

        
        clen = len(px)
                #img_piece = this_gray[:,this_pathx[0]:this_pathx[clen-1]]
                # no crop blank version 
        #factor=W2/W
         
        this_pathy = py 
        #resample 
         
        # first determine the lef piece
        pathl = np.zeros(int(px[0]))+ H-1
        len1 = len(this_pathy)
        len2 = len(pathl)
        pathr = np.zeros(W-len1-len2) + H-1
        path_piece = np.append(pathl,this_pathy,axis=0)
        path_piece = np.append(path_piece,pathr,axis=0)
        new_y = signal.resample(path_piece, W)

        
        
        #in the down sample pathy function the dot with no label will be add a number of Height, 
        #however because the label software can not label the leftmost and the rightmost points,
        #so it will be given a max value,  I crop the edge of the label, remember to crop the image correspondingly .
        new_x = np.arange(0, W)
        #path_piece = signal.resample(path_piece[3:W2-3], W2)
        return new_x,new_y