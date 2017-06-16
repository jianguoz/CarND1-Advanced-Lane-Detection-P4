'''Writen by Jianguo Zhang, June 14, 2017.
   jzhan51@uic.edu
'''

import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
#% matplotlib inline

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, n):
        # How many frames to use to smooth
        self.n = n
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        
        # Line smoothing
        # Second order polynomial curve, f(y) = A*y**2+B*y+c
        self.A = []
        self.B = []
        self.C = []
        self.ave_A = 0
        self.ave_B = 0
        self.ave_C = 0
        
        # Smooth curvature
        self.curv = []
        #self.right_curv = []
        self.ave_curv = 0
        #self.ave_right_curv = 0
        
    def add_to_smooth_fit(self, polynomial_fit):
        '''Smoothing over the last n frames'''
        
        self.A.append(polynomial_fit[0])
        self.B.append(polynomial_fit[1])
        self.C.append(polynomial_fit[2])
        
        # Keep size of A|B|C equal to n
        if len(self.A)>self.n:
            self.A.pop(0)
            self.B.pop(0)
            self.C.pop(0)
        
        self.ave_A = np.mean(self.A)
        self.ave_B = np.mean(self.B)
        self.ave_C = np.mean(self.C)
        
        return self.ave_A, self.ave_B, self.ave_C   
    
    def ave_curvature(self, curvature):
        '''For a frame without detected lanes, 
           we assign average curvature to that frame'''
        
        self.curv.append(curvature)
        #self.right_curv.append(right_curvature)
        
        # Keep size of left_curv|rightcurv equal to n
        if len(self.curv)>self.n:
            self.curv.pop(0)
        #    self.right_curv.pop(0)
            
        self.ave_curv = np.mean(self.curv)
        #self.ave_right_curv = np.mean(self.right_curv)
        
        return self.ave_curv
    
    def get_results_of_smooth_fit(self):
        return self.ave_A, self.ave_B, self.ave_C
    
    def get_results_of_ave_curvature(self):
        return self.ave_curv