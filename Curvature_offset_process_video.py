'''Writen by Jianguo Zhang, June 14, 2017.
   jzhan51@uic.edu
'''

import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import pickle
from Line_class import Line
from combined_thresh import combined_thresh
from Perspective_transform import perspective_transorm
from polynomial_fit import line_fit, advanced_fit, line_fit_visualize, advanced_fit_visualize
#% matplotlib inline

def calculate_curvature(leftx, rightx, lefty, righty):
    '''Calculate the radius of curvature in meters'''
    
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    #y_eval = np.max(ploty)
    y_eval = 719
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    
    return left_curverad, right_curverad

def calculate_offset(undist, left_fit, right_fit):
    '''Calculate the offset of the lane center from the center of the image'''
    
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    ploty = undist.shape[0] # height
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    offset = (left_fitx+right_fitx)/2 - undist.shape[1]/2 # width 
    offset = xm_per_pix*offset
    
    return offset

def final_drawing(undist, left_fit, right_fit, left_curverad, right_curverad, Minv, vehicle_offset):
    '''Project the measurement back down onto the original undistorted image of the road'''
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw the lines on
    warped = np.zeros((720,1280))
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    # Anotate curvature values 
    ave_curvature = (left_curverad + right_curverad)/2
    ave_text = 'Radius of average curvature: %.2f m'%ave_curvature
    cv2.putText(result, ave_text,(50,50), 0, 1, (0,0,0), 2, cv2.LINE_AA)
    
    # Anotate vehicle offset from the lane center
    if(vehicle_offset>0):
        offset_text = 'Vehicle right offset from lane center: {:.2f} m'.format(vehicle_offset)
    else:
        offset_text = 'Vehicle left offset from the lane center: {:.2f} m'.format(-vehicle_offset)
    cv2.putText(result, offset_text,(50,80), 0, 1, (0,0,0), 2, cv2.LINE_AA)
    
    #plt.imshow(result)
    
    return result

def process_video_image(img):
    '''Process each image in the video and return with annotated image'''
    
    global detected, mtx, dist, left_lanes, right_lanes
    # Undistorting image
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Combined thresh
    combined = combined_thresh(undist)

    # Warped image
    binary_warped, _, Minv = perspective_transorm(combined)

    # Line fit
    if not detected:
        curv_pickle = line_fit(binary_warped)
        
        if curv_pickle is not None:
            left_fit = curv_pickle["left_fit"]
            right_fit = curv_pickle["right_fit"]
            leftx = curv_pickle["leftx"]
            lefty = curv_pickle["lefty"]
            rightx = curv_pickle["rightx"]
            righty = curv_pickle["righty"]

            # Update
            # Add into the set to smoothing average fit
            left_fit = left_lanes.add_to_smooth_fit(left_fit)
            right_fit = right_lanes.add_to_smooth_fit(right_fit)

            # Calculate curvature
            # Assume first frame can be detected both lanes and curvatures 
            left_curvature, right_curvature = calculate_curvature(leftx, rightx, lefty, righty)

#             # Add into the set to smoothing average curvature
#             # Only use for a frame without detected lanes
#             special_left_curv = left_lanes.ave_curvature(left_curvature)
#             special_right_curv = right_lanes.ave_curvature(right_curvature)

            detected = True
        else:
            # Calculate based on previous frames
            left_fit = left_lanes.get_results_of_smooth_fit()
            right_fit = right_lanes.get_results_of_smooth_fit()
            special_left_curv = left_lanes.get_results_of_ave_curvature()
            special_right_curv = right_lanes.get_results_of_ave_curvature()
            left_curvature, right_curvature = special_left_curv, special_right_curv
            detected = False
    else:
        # Smooth fit
        left_fit = left_lanes.get_results_of_smooth_fit()
        right_fit = right_lanes.get_results_of_smooth_fit()
        
        # Add into the set to smoothing average curvature
        # Only use for a frame without detected lanes
        special_left_curv = left_lanes.get_results_of_ave_curvature()
        special_right_curv = right_lanes.get_results_of_ave_curvature()
        
        # Skip the sliding windows step once you know where the lines are
        # Search in a margin around the previous line position 
        curv_pickle = advanced_fit(binary_warped, left_fit, right_fit)
        
        if curv_pickle is not None:
            # Detected lines 
            left_fit = curv_pickle["left_fit"]
            right_fit = curv_pickle["right_fit"]
            leftx = curv_pickle["leftx"]
            lefty = curv_pickle["lefty"]
            rightx = curv_pickle["rightx"]
            righty = curv_pickle["righty"]
            
            # Note: We only make update when detect lanes in the current frame
            # Add into the set to smoothing average fit
            left_fit = left_lanes.add_to_smooth_fit(left_fit)
            right_fit = right_lanes.add_to_smooth_fit(right_fit)
            
            # Calculate curvature
            left_curvature, right_curvature = calculate_curvature(leftx, rightx, lefty, righty)
        else:
            # Calculate curvature based on previous frames
            left_curvature, right_curvature = special_left_curv, special_right_curv
            detected = False
            
        
    # Calculate vehicle offset from the lane center in the image
    vehicle_offset = calculate_offset(undist, left_fit, right_fit)
    if vehicle_offset > 0.25:
        detected = False
    
    # Project the measurement back down onto the original undistorted image of the road
    result = final_drawing(undist, left_fit, right_fit, left_curvature, right_curvature, Minv, vehicle_offset)
    
    return result


if __name__ == '__main__':

    # Read in the saved camera matrix and distortion coefficients
    dist_pickle = pickle.load(open("./camera_cal/camera_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

     # Define number of frames for smoothing
    num_frames = 5
    left_lanes = Line(n=num_frames) 
    right_lanes = Line(n=num_frames)
    detected = False 

    img = mpimg.imread('./test_images/test2.jpg')
    result = process_video_image(img)
    plt.imshow(result)

    mpimg.imsave('./output_img/test2_final_projected.jpg', result)

    # Visualize images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(result)
    ax2.set_title('Polynomial Fit Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig('./output_img/test2_final_projected_example.jpg')   

    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip
    from IPython.display import HTML

    # Define number of frames for smoothing
    num_frames = 8
    left_lanes = Line(n=num_frames) 
    right_lanes = Line(n=num_frames)
    detected = False 

    white_output = 'project_demo.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_video_image) #NOTE: this function expects color images!!
    #%time white_clip.write_videofile(white_output, audio=False)
    white_clip.write_videofile(white_output, audio=False)    
