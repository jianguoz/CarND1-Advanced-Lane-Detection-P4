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
from combined_thresh import combined_thresh
#% matplotlib inline

def perspective_transorm(img):
    
    # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
    # Note: you could pick any four of the detected corners 
    # as long as those four corners define a rectangle
    # VOne especially smart way to do this would be to use four well-chosen
    # corners that were automatically detected during the undistortion steps
    # We recommend using the automatic detection of corners in your code
    # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    # e) use cv2.warpPerspective() to warp your image to a top-down view           
    
    offset = 250
    # Draw corners
    #cv2.drawChessboardCorners(undist, (nx,ny), corners, ret)
   # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    gray=img
    # Grab the image size (width, height)
    img_size = (gray.shape[1], gray.shape[0])
        
    src = np.float32([[200,img_size[1]], [1200, img_size[1]],\
                      [700,450], [585,450]])
    dst = np.float32([[300, img_size[1]],[960, img_size[1]],\
                      [960, 0],[300, 0]])
    
    #src = np.float32([[gray.shape[1]/2, gray.shape[0]/2-offset],[gray.shape[1], offset],[gray.shape[1], gray.shape[0]-offset],[gray.shape[1]/2, gray.shape[0]/2+offset]])
    #dst = np.float32([[300, 720],[980, 720],[300, 0],[980, 0]])
        
    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    
    # Calculate the inverse perspective matrix (Minv)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    
        
    # Warp the image
    warped = cv2.warpPerspective(gray, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, M, M_inv


if __name__ == '__main__':

    # Read in the saved camera matrix and distortion coefficients
    dist_pickle = pickle.load(open("./camera_cal/camera_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Take test2.jpg as an example
    img = mpimg.imread('./test_images/test2.jpg')
    #img = mpimg.imread('./output_img/test6_threshold.jpg')
    nx = 9
    ny = 6
    #undist = combined_thresh(img)

    # Undistorting image
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Combined thresh
    combined = combined_thresh(undist)

    # top_down also mean binary_warped
    top_down, perspective_M, M_inv = perspective_transorm(combined)
    mpimg.imsave('./output_img/test2_binary_warped_gray.jpg', top_down, cmap='gray')
    # Visualize images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(top_down, cmap='gray')
    ax2.set_title('Undistorted and Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig('./output_img/test2_binary_warped_example.jpg')