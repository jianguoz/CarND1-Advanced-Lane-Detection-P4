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

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    '''Calculated directional gradient'''
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Take the derivative in x or y given orient = 'x' or 'y'
    if(orient=='x'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if(orient=='y'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Take the absolute value
    abs_sobel = np.absolute(sobel)
    
    # Scale to 8 bit(0-255) then convert to type=np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    #Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel>thresh[0]) & (scaled_sobel<thresh[1])] = 1
    
    return sbinary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0,255)):
    '''Calculate gradient magnitude'''
   
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Take the derivative in x and y 
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Take the absolute value
    abs_sobel = np.sqrt(sobelx**2+sobely**2)
    
    # Scale to 8 bit(0-255) then convert to type=np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    #Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel>=mag_thresh[0]) & (scaled_sobel<=mag_thresh[1])] = 1
    
    return sbinary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    '''Calculate gradient '''
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Take the derivative in x and y 
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Take the absolute value of the x and y gradients
    sobelx = np.absolute(sobelx)
    sobely = np.absolute(sobely)
    
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    arc_results = np.arctan2(sobely, sobelx)
    
#     # Scale to 8 bit(0-255) then convert to type=np.uint8
#     arc_results = np.uint8(255*arc_results/np.max(arc_results))
    
    # Create a binary mask where direction thresholds are met
    binary_out = np.zeros_like(arc_results)
    binary_out[(arc_results>thresh[0]) & (arc_results<thresh[1])] = 1
    
    return binary_out

def hls_color(img, thresh=(0,255)):
    '''We already saw that standard grayscaling lost color information for
       the lane lines. So using hls color channel'''
    
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    
    # Scale to 8 bit(0-255) then convert to type=np.uint8
    #S = np.uint8(255*S/np.max(S))
    
    # Apply a mask to S channel
    binary_out = np.zeros_like(S)
    binary_out[(S>thresh[0]) & (S<=thresh[1])] = 1
    
    return binary_out

def yuv_color(img, thresh=(0,255)):
    '''Another effective approach is to obtain lane pixels by color. 
       The rationale behind it is that lanes in this project are either yellow or white. 
       we can also only YUV color space to do the thresholding, 
       Pixels with a V component less than 105 are deemed white, while pixels with a Y
       component greater than or equal to 205 are deemed yellow'''
    
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    Y = yuv[:,:,0]
    U = yuv[:,:,1]
    V = yuv[:,:,2]
    
    # Apply a mask to S channel
    binary_out = np.zeros_like(Y)
    binary_out[(Y>=thresh[1]) & (Y<=255)] = 1
    binary_out[(V>0) & (V<thresh[0])] = 1
    
    return binary_out

def combined_thresh(image):
    # Apply each of the thresholding functions
    ksize = 3
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(80, 255))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(80, 255))
    mag_binary = mag_thresh(image, sobel_kernel=9, mag_thresh=(100, 255))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
    s_binary = hls_color(image, thresh=(170, 255))
    yuv_binary = yuv_color(image, thresh=(105, 205))

    combined = np.zeros_like(dir_binary)
    combined[(gradx ==1| ((mag_binary == 1)&(dir_binary == 1))) | yuv_binary == 1] = 1
   # combined = yuv_binary
    return combined


if __name__ == '__main__':

    # Read in the saved camera matrix and distortion coefficients
    dist_pickle = pickle.load(open("./camera_cal/camera_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Choose a Sobel kernel size
    # Taking the gradient over larger regions can smooth over noisy intensity fluctuations on small scales. 
    # Choose a larger odd number to smooth gradient measurements
    ksize = 3 

    # Take test2.jpg as an example
    image = mpimg.imread('./test_images/test2.jpg')

    # Undistorting image
    dst = cv2.undistort(image, mtx, dist, None, mtx)

    combined = combined_thresh(dst)
    # Save warped image
    mpimg.imsave('./output_img/test2_threshold.jpg', combined, cmap='gray')
    #mpimg.imsave('./output_img/test2_threshold.jpg', combined)
    # Plotting thresholded images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.set_title('Stacked thresholds')
    ax1.imshow(image)

    ax2.set_title('Combined S channel and gradient thresholds')
    ax2.imshow(combined, cmap='gray')
    plt.savefig('./output_img/test2_threshold_example.jpg')