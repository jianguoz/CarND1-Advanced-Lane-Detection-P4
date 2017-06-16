def calculate_curvature(ploty, leftx, rightx):
    '''Calculate the radius of curvature in meters'''
    
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    
    return left_curverad, right_curverad

def calculate_offset(undist, left_fit, right_fit):
    '''Calculate the offset of the lane center from the center of the image'''
    
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    ploty = undist.shape[0]
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    offset = (left_fitx+right_fitx)/2 - ploty/2
    offset = xm_per_pix*offset
    
    return offset

def final_drawing(undist, left_fitx, right_fitx, left_curverad, right_curverad, Minv, vehicle_offset):
    '''Project the measurement back down onto the original image of the road'''
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    # Anotate curvature values 
    ave_curvature = (left_curverad + right_curverad)/2
    ave_text = 'Radius of average curvature: %.2f m'%ave_curvature
    cv2.putText(result, ave_text,(50,50), 0, 1, (0,0,0), 2, cv2.LINE_AA)
    
    # Anotate vehicle offset from the lane center
    if(vehicle_offset>0):
        offset_text = 'Vehicle right offset from lane center: {%.2f} m'.format(vehicle_offset)
    else:
        offset_text = 'Vehicle left offset from the lane center: {%.2f} m'.format(-vehicle_offset)
    cv2.putText(result, offset_text,(50,80), 0, 1, (0,0,0), 2, cv2.LINE_AA)
    
    plt.imshow(result)
    
    return result