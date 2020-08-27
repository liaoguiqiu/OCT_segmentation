import cv2
import math
import numpy as np
import os
import random
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from operater import Basic_Operator
class A_line_syn():
        
    def __init__(self):
        self.plot_flag   = True
        self.pixnum = 1024
        #boundary position
        self.bound_pos=np.array([300,600,700]) # the first layer is air
        self.draw_fig_flag = False # the flag for analy drawing
        self.y=[]
        # import the draw signal package
    def generator_Aline(self,H,bound_pos): # input bound psition should be a array and the max position of the array should be = H
        bound_pos=np.array(bound_pos) 
        bound_pos = bound_pos.astype(int)
        u0 = -1/500 # this is the basic attenuation coeff
        uii = -1/900
        I0 = 400 # this is the basic  intensity from the beginning 
        num = len(bound_pos)
        this_bound_pos  = bound_pos
        blank_intensity= 30+30* np.random.random_sample()
        signals = np.zeros((H,num)) # buffer every piece
        windows = np.zeros((H,num)) # buffer every window
        #T_d  =(((x-_focal_point)/Rayleighrange ).^2 +1).^(-0.5)
        x2 =  np.arange(0, H)  
        focal_point  = 50
        Rayle = 20 
        squre = np.square((x2-focal_point)/Rayle/20)
        effct = np.power(squre+1,-0.5)
        # the signal are deal simaarly in each peiece 
        for i in range(num):
                f  = 1+2*i* np.random.random_sample()
                ui  = f * u0 # this coefficient 
                x = np.arange(0, H) - bound_pos[i]
                #Ii = I0 * ( 1+  2*(bound_pos[i] - bound_pos[0] )/H)
                Ii = I0 * np.exp((bound_pos[i] - bound_pos[0] ) * uii)
                signals[:,i] = Ii* np.exp(x * ui)
                # add  the focusing effect 
                

                signals[:,i] = np.multiply (signals[:,i], effct)



        # window needs to addressed with differemt situation
        # the last window is always the same 
        windows[:,num-1] = np.append(np.zeros(this_bound_pos[num-1]),np.ones(H-this_bound_pos[num-1]))


        if num > 1:  # the last piece is special : will decay till edge, other situation
            for i in range(num-1):
                this_win =   np.append(np.zeros(this_bound_pos[i]),np.ones(this_bound_pos[i+1]-this_bound_pos[i]))
                this_win = np.append(this_win,np.zeros(H-this_bound_pos[i+1]))
                windows[:,i] =  this_win
            # deal the last piece :
        output = np.zeros(H)  
        # sum all with windows 
        for i in range(num):
            this_y = signals[:,i] 
            this_win =  windows[:,i] 
            output = output + np.multiply ( this_y , this_win) 

        # the intensity  of the blank area before the tissue
        blank = np.append(blank_intensity * np.ones(this_bound_pos[0]),np.zeros(H-this_bound_pos[0]))
        output = output + blank
        return output,effct

        #this_bound_pos  = self.bound_pos - shift
        ## a small range of the attenuation u0
        #0.05+0.91* np.random.random_sample()
        #x = np.arange(0, H)
        #y1 = np.exp(-x/300)
        #y2  = 2*np.exp(-x/300)
        #y3 = 3*np.exp(-x/300)
        #window1 = np.append(np.zeros(this_bound_pos[0]),np.ones(this_bound_pos[1]-this_bound_pos[0]))
        #window1 = np.append(window1,np.zeros(H-this_bound_pos[1]))
        #window2 = np.append(np.zeros(this_bound_pos[1]),np.ones(this_bound_pos[2]-this_bound_pos[1]))
        #window2 = np.append(window2,np.zeros(H-this_bound_pos[2]))
        #window3 = np.append(np.zeros(this_bound_pos[2]),np.ones(H-this_bound_pos[2]))

        #y  =  np.multiply ( y1 , window1) + np.multiply ( y2 , window2) + np.multiply ( y3 , window3)

        #self.y  = y 
        #return output
    def plot_1d(self,y):
        if self.plot_flag == True:
     
            plt.plot(y)
            pass
        pass
    def fake_image(self,H,W,contourx, contoury):
        image  = np.zeros((H,W))
        for i in range(W):
             
            source_line,effect = self.generator_Aline(H,[contoury[0][i],contoury[1][i],contoury[2][i],contoury[3][i]])  

            #new[:,i] = ba.warp_padding_line1(source_line, contour0y[line_it],new_contoury[i])
            image[:,i] = source_line
            #self.plot_1d(source_line)
            #random select a source
            pass
        pass
        return image
    def random_image(self,H,W):
        image  = np.zeros((H,W))
        for i in range(W):
             
            source_line,effect = self.generator_Aline(H,[300,600,900])  

            #new[:,i] = ba.warp_padding_line1(source_line, contour0y[line_it],new_contoury[i])
            image[:,i] = source_line
            #random select a source
            pass
        pass
        return image

if __name__ == '__main__':
    generator = A_line_syn()
    bound_pos=np.array([400,600,800])
    y ,effect = generator.generator_Aline(1024, bound_pos)
    generator.plot_1d(y)
    generator.plot_1d(effect)


    one_frame = generator.random_image(1024,832)
    kernel = np.ones((5,5),np.float32)/25
    one_frame = cv2.filter2D(one_frame,-1,kernel)
    #one_frame = cv2.GaussianBlur(one_frame,(5,5),0)
    one_frame  = Basic_Operator . noisy( "gauss_noise" ,  one_frame )
    one_frame  = Basic_Operator . noisy( "s&p" ,  one_frame )

    one_frame  = Basic_Operator . noisy( "poisson" ,  one_frame )
    #one_frame  = Basic_Operator . noisy( "speckle" ,  one_frame )


    cv2.imshow('color',one_frame.astype( np.uint8))
    cv2.waitKey(10)   



         