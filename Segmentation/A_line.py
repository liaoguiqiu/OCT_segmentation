import os
import math
import numpy as np
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
class A_line_process():
        
    def __init__(self):
        self.draw_fig_flag = False # the flag for analy drawing
        # import the draw signal package

         
    def find_4peak(self, x):
        if self.draw_fig_flag == True:      
            import matplotlib.pyplot as plt
        W = x.size
        border1 = np.zeros(W) +20
        border2= np.ones(W)*1000
        peaks, _ = find_peaks(x, height=(border1, border2),distance=50,prominence=1)
        if self.draw_fig_flag == True:
            plt.plot(x)
            plt.plot(border1, "--", color="gray")
            plt.plot(border2, ":", color="gray")
            plt.plot(peaks, x[peaks], "x")
            plt.show()
        return peaks
#test 
if __name__ == '__main__':
    A_line = A_line_process()
    if A_line.draw_fig_flag == True:      
            import matplotlib.pyplot as plt
    x = electrocardiogram()[2000:4000]
    peaks, _ = find_peaks(x, height=0)
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.plot(np.zeros_like(x), "--", color="gray")
    plt.show()