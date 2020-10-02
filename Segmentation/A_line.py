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
        peaks, _ = find_peaks(x, height=(border1, border2),distance=15,prominence=1)
        peak_num  = len(peaks)
        peakvalue = x[peaks]
        hight_index  = np.argsort((-1)*peakvalue)
        peaks_new = peaks
        for i in  range( peak_num):
            peaks_new[i] = peaks[hight_index[i]]
        ## rechanege the  sequence select 4 for based on the peak value 
        #peaks = [peaks[hight_index[0]], 
        #         peaks[hight_index[1]],
        #         #peaks[hight_index[2]],
        #         #peaks[hight_index[3]],
        #         #peaks[hight_index[4]]
        #         ]
        # sort agian based on the index value 
        #sort_index  = np.argsort(peaks[0:2])
        #peaks = [peaks[sort_index[0]], 
        #         peaks[sort_index[1]],
        #         #peaks[sort_index[2]],
        #         #peaks[3],
        #         #peaks[hight_index[4]]
        #         ]

        #peaks =np.sort(peaks)
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