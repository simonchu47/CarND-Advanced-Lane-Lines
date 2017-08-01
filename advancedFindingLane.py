#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:32:27 2017

@author: simon
"""

import os
import pickle
import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import random
from moviepy.editor import VideoFileClip
from collections import deque

flags = tf.app.flags
FLAGS = flags.FLAGS

# Command line flags
flags.DEFINE_string('calibration_path', '', "The directory containing camera calibration images")
flags.DEFINE_bool('calibration_camera', False, "Calibration the camera or not")
flags.DEFINE_integer('x_corners', 9, "The number of inside corners in x")
flags.DEFINE_integer('y_corners', 6, "The number of inside corners in y")
flags.DEFINE_bool('image', False, "Input is image")
flags.DEFINE_bool('video', False, "Input is video")
flags.DEFINE_string('image_path', '', "The input image file path")
flags.DEFINE_string('video_path', '', "The input video file path")
flags.DEFINE_bool('perspective', False, "Generate perspective transform matrix")

class Line():
    def __init__(self):
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
    def check_last_detected(self):
        return self.detected
left_line = Line()
right_line = Line()
    
def grayscale(img):
    # Applies the Grayscale transform    
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Or use RGB2GRAY if you read an image with mpimg.imread()
    # return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    image_shape = img.shape
    right_lines_slope = []
    right_lines_end = [image_shape[1], 0]
    left_lines_slope = []
    left_lines_end = [0,0]
    for line in lines:
        for x1,y1,x2,y2 in line:
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            line_slope = (y2-y1)/(x2-x1)
            line_slope_abs = abs(line_slope)
            if line_slope < 0 and line_slope_abs > 0.3 and line_slope_abs < 1:
                left_lines_slope.append([(y2-y1)/(x2-x1)])
                if y1 > y2:
                    if x1 > left_lines_end[0]:
                        left_lines_end = [x1, y1]
                else:
                    if x2 > left_lines_end[0]:
                        left_lines_end = [x2, y2]
            elif line_slope > 0 and line_slope_abs > 0.3 and line_slope_abs < 1:
                right_lines_slope.append([(y2-y1)/(x2-x1)])
                if y1 > y2:
                    if x1 < right_lines_end[0]:
                        right_lines_end = [x1, y1]
                else:
                    if x2 < right_lines_end[0]:
                        right_lines_end = [x2, y2]
    
    plot_hight_ratio_top = 0.63
    plot_hight_ratio_bot = 0.97
    if not len(right_lines_slope) == 0:
        average_right_lines_slope = np.mean(right_lines_slope)
        right_end1 = (int((image_shape[0]*plot_hight_ratio_bot-right_lines_end[1])/average_right_lines_slope + right_lines_end[0]), int(image_shape[0]*plot_hight_ratio_bot))
        right_end2 = (int((image_shape[0]*plot_hight_ratio_top-right_lines_end[1])/average_right_lines_slope + right_lines_end[0]), int(image_shape[0]*plot_hight_ratio_top))
        cv2.line(img, right_end1, right_end2, color, thickness)
        #print("average_right_lines_slope: %.2f" %average_right_lines_slope)
        #print("right_lines_end: (%d, %d)" %(right_lines_end[0], right_lines_end[1]))
        #print("right_end: (%d, %d), (%d, %d)" %(right_end1[0], right_end1[1], right_end2[0], right_end2[1]))
    if not len(left_lines_slope) == 0:
        average_left_lines_slope = np.mean(left_lines_slope)
        left_end1 = (int((image_shape[0]*plot_hight_ratio_bot-left_lines_end[1])/average_left_lines_slope + left_lines_end[0]), int(image_shape[0]*plot_hight_ratio_bot))
        left_end2 = (int((image_shape[0]*plot_hight_ratio_top-left_lines_end[1])/average_left_lines_slope + left_lines_end[0]), int(image_shape[0]*plot_hight_ratio_top))
        cv2.line(img, left_end1, left_end2, color, thickness)
        #print("average_left_lines_slope: %.2f" %average_left_lines_slope)
        #print("left_lines_end: (%d, %d)" %(left_lines_end[0], left_lines_end[1]))
        #print("left_end: (%d, %d), (%d, %d)" %(left_end1[0], left_end1[1], left_end2[0], left_end2[1]))
    
    # For source points I'm grabbing the outer four detected corners
    src_points = np.float32([left_end2, right_end2, right_end1, left_end1])
    #src_points = np.array([right_end1,right_end2,left_end1,left_end2], dtype=np.float)
    dst_points = np.float32([(left_end1[0], 0), (right_end1[0], 0), (right_end1[0], image_shape[0]), (left_end1[0], image_shape[0])])
    #dst_points = np.array([right_end1,(right_end1[0], right_end2[1]),left_end1,(left_end1[0], left_end2[1])], dtype=np.float)
    points_pickle = {}
    points_pickle["src_points"] = src_points
    points_pickle["dst_points"] = dst_points
    return points_pickle

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    points = draw_lines(line_img, lines)
    return line_img, points

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)
    
def gen_perspective_trans_matrix(image):
    #Grayscale the image
    gray = grayscale(image)
    
    #Define a kernel size and apply Gaussian smoothing
    kernel_size = 3
    blur_gray = gaussian_blur(gray, kernel_size)
    
    #Define canny parameters and apply it to find edges
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    
    #Define a mask for edges of interesting region
    image_shape = image.shape
    left_top = (image_shape[1]*28/64,image_shape[0]*10/16)
    left_bot = (0,image_shape[0])
    right_top = (image_shape[1]*36/64,image_shape[0]*10/16)
    right_bot = (image_shape[1],image_shape[0])
    vertices = np.array([[left_bot,left_top,right_top,right_bot]], dtype=np.int32)
    masked_edge = region_of_interest(edges, vertices)
    
    #Define Hough transform parameters and apply it find the lane lines
    rho = 2
    theta = np.pi/180
    threshold = 45
    min_line_length = 5
    max_line_gap = 5
    line_image, points = hough_lines(masked_edge, rho, theta, threshold, min_line_length, max_line_gap)
    src = points["src_points"]
    dst = points["dst_points"]
    #print("src points is {}".format(src))
    #print("dst points is {}".format(dst))
    M = cv2.getPerspectiveTransform(src, dst)
    rM = cv2.getPerspectiveTransform(dst, src)
    
    #img_size = (image.shape[1], image.shape[0])
    #mixed_image = weighted_img(line_image, image)
    #warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    #cv2.imwrite('./perspective_lines.jpg', mixed_image)
    #cv2.imwrite('./perspective_trans.jpg', warped)
    return M, rM    

def hls_filter(image, h_bottom, h_upper, l_bottom, l_upper, s_bottom, s_upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
        
    filtered_h = np.zeros_like(h_channel, dtype=np.bool)
    filtered_h[(h_channel<h_upper) & (h_channel>h_bottom)] = True
        
    filtered_s = np.zeros_like(s_channel, dtype=np.bool)
    filtered_s[(s_channel<s_upper) & (s_channel>s_bottom)] = True
        
    filtered_l = np.zeros_like(l_channel, dtype=np.bool)
    filtered_l[(l_channel<l_upper) & (l_channel>l_bottom)] = True

    #combined =  np.zeros_like(image)
    condition = filtered_h & filtered_s & filtered_l
    #combined[condition] = image[condition]
    
    #return combined
    return condition
    

def sliding_windows_finding_lines(binary_warped, xm_per_pix, ym_per_pix):
    # Assuming a warped binary image called "binary_warped" is created
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    #out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    if (len(leftx)>0) and (len(lefty)>0):
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    else:
        left_fit_cr = []
    
    if (len(rightx)>0) or (len(righty)>0):
        right_fit = np.polyfit(righty, rightx, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    else:
        right_fit_cr = []
    
    # Fit new polynomials to x,y in world space
    #left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    #right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # The bottom of the image 
    y_eval = binary_warped.shape[0] - 1
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    print("left_curvature is {}".format(left_curverad))
    print("right_curvature is {}".format(right_curverad))
    
    left_fit_x_pos = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_fit_x_pos = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    
    """
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.savefig("./test1_sliding_windows.jpg")
    """
    
    return left_fit, right_fit, left_curverad, right_curverad, left_fit_x_pos, right_fit_x_pos

def known_fit_finding_lines(binary_warped, left_fit, right_fit, xm_per_pix, ym_per_pix):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    
    # Fit a second order polynomial to each    
    if (len(leftx)>0) and (len(lefty)>0):
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        lefty=np.linspace(0, binary_warped.shape[0]-1, num=binary_warped.shape[0])
        leftx = np.array([left_fit[0]*y**2 + left_fit[1]*y + left_fit[2] 
                              for y in lefty])
        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    
    if (len(rightx)>0) or (len(righty)>0):
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        righty=np.linspace(0, binary_warped.shape[0]-1, num=binary_warped.shape[0])
        rightx = np.array([right_fit[0]*y**2 + right_fit[1]*y + right_fit[2] 
                              for y in righty])
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # The bottom of the image 
    y_eval = binary_warped.shape[0] - 1
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    left_fit_x_pos = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_fit_x_pos = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    
    """
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    #plt.savefig("./test1_known_fit.jpg")
    """
    
    return left_fit, right_fit, left_curverad, right_curverad, left_fit_x_pos, right_fit_x_pos

def draw_lane(image, left_fit, right_fit, Minv):
    # Create an image to draw the lines on
    #warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    color_warp = np.zeros_like(image).astype(np.uint8)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    
    return result
        
# The following is to process the calibration of camera
# Decided by the command flags
nx = FLAGS.x_corners
ny = FLAGS.y_corners

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.


if FLAGS.calibration_camera is True:
    if FLAGS.calibration_path is not "":
        cali_path = FLAGS.calibration_path + '/'
        cali_files = []
        for x in os.listdir(cali_path):
            cali_files.append(cali_path+x)
            img = cv2.imread(cali_path+x)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret is True:
                objpoints.append(objp)
                imgpoints.append(corners)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
        cv2.destroyAllWindows()
        
        # Test undistort an image from the calibration images directory
        test_img_file = random.choice(cali_files)
        print("To test undistorting image: " + test_img_file + "..............")
        img = cv2.imread(test_img_file)
        img_size = (img.shape[1], img.shape[0])
        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imshow('undistorted_img', dst)
        #cv2.imwrite('./test_undist.jpg',dst)
        
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump( dist_pickle, open( "./camera_calibration_pickle.p", "wb" ) )

    else:
         print("Please add the calibration_path flag...")       
    #print(cali_files)
else:
    # Read in the camera matrix and distortion coefficients
    camera_calibration_pickle = "./camera_calibration_pickle.p"
    if os.path.isfile(camera_calibration_pickle) is True:
        print("Find the pickle.........")
        camera_pickle = pickle.load(open(camera_calibration_pickle, "rb"))
        mtx = camera_pickle["mtx"]
        dist = camera_pickle["dist"]
    else:
        print("Sorry!, no camera calibration pickle...")

if FLAGS.perspective is True:
    if FLAGS.image_path is not "":
        if os.path.isfile(FLAGS.image_path) is True:
            img = cv2.imread(FLAGS.image_path)
            undist = cv2.undistort(img, mtx, dist, None, mtx)
            M, rM = gen_perspective_trans_matrix(undist)
            perspective_trans_pickle = {}
            perspective_trans_pickle["M"] = M
            perspective_trans_pickle["rM"] = rM
            pickle.dump(perspective_trans_pickle, open( "./perspective_trans_pickle.p", "wb" ) )
        else:
            print("There is no input image...")
    else:
        print("Please add the input image path...")
else:
    # Read in the perspective transformation matrix
    perspective_trans_pickle_file = "./perspective_trans_pickle.p"
    if os.path.isfile(perspective_trans_pickle_file) is True:
        print("Find the perspective transormation pickle.........")
        perspective_trans_pickle = pickle.load(open(perspective_trans_pickle_file, "rb"))
        M = perspective_trans_pickle["M"]
        rM = perspective_trans_pickle["rM"]
    else:
        print("Sorry!, no perspective transormation pickle...")

# The Pipeline
#def pipeline(image, s_thresh=(170, 255), h_g_thresh=(20, 200), r_thresh=(20, 100)):
def pipeline(image, h_thresh=(98.0, 102.0), l_thresh=(200.0, 255.0), s_thresh=(170.0, 255.0), sobel_thresh=(80.0, 255.0)):
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Performs image distortion correction
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    
    """
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(h_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    sobely = cv2.Sobel(h_channel, cv2.CV_64F, 0, 1) # Take the derivative in y
    sobel_mag = np.sqrt(np.square(sobelx)+np.square(sobely))
    #abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*sobel_mag/np.max(sobel_mag))
    
    # Threshold x gradient
    hgbinary = np.zeros_like(scaled_sobel)
    hgbinary[(scaled_sobel >= h_g_thresh[0]) & (scaled_sobel <= h_g_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    r_channel = undist[:,:,0]
    sobel_r = cv2.Sobel(r_channel, cv2.CV_64F, 1, 1)
    abs_sobelr = np.absolute(sobel_r)
    scaled_sobelr = np.uint8(255*abs_sobelr/np.max(abs_sobelr))
    r_sobel_binary = np.zeros_like(scaled_sobelr)
    r_sobel_binary[(scaled_sobelr>=r_thresh[0]) & (scaled_sobelr<=r_thresh[1])] = 1
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    #color_binary = np.dstack((r_sobel_binary , sxbinary, s_binary))
    combined_binary = np.zeros_like(scaled_sobelr)
    #combined_binary[(r_sobel_binary==1)|(lgbinary==1)|(s_binary==1)] = 255
    #combined_binary[(lgbinary==1)|(s_binary==1)] = 255
    #combined_binary[(s_binary==1)] = 255
    #combined_binary[(hgbinary==1)] = 255
    combined_binary[(r_sobel_binary==1)] = 255
    #combined_binary[(r_sobel_binary==1)|(s_binary==1)] = 255
    """
    #hls_filtered = hls_filter(undist, h_thresh[0], h_thresh[1], l_thresh[0], l_thresh[1], s_thresh[0], s_thresh[1])
    yellow_line_filter = hls_filter(undist, 90.0, 101.0, 0.0, 255.0, 50.0, 255.0)
    white_line_filter = hls_filter(undist, 0.0, 180.0, 210.0, 255.0, 0.0, 255.0)
    hls_filtered = np.zeros_like(undist)
    all_filter = yellow_line_filter | white_line_filter
    #all_filter = yellow_line_filter
    hls_filtered[all_filter] = undist[all_filter]
    gray_filtered = cv2.cvtColor(hls_filtered, cv2.COLOR_BGR2GRAY)
    
    """
    sobel = cv2.Sobel(gray_filtered, cv2.CV_64F, 1, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel>=sobel_thresh[0]) & (scaled_sobel<=sobel_thresh[1])] = 255
    """
    
    combined_binary = np.zeros_like(gray_filtered)
    combined_binary[gray_filtered>0] = 255
    warped = cv2.warpPerspective(combined_binary, M, (combined_binary.shape[1], combined_binary.shape[0]), flags=cv2.INTER_LINEAR)                   
    
    if left_line.check_last_detected() == True and left_line.check_last_detected() == True:
        left_fit = left_line.current_fit
        right_fit = right_line.current_fit
        left_fit, right_fit, left_line.radius_of_curvature, right_line.radius_of_curvature, left_fit_x_pos, right_fit_x_pos = known_fit_finding_lines(warped, left_fit, right_fit, xm_per_pix, ym_per_pix)
        left_line.current_fit = left_fit
        right_line.current_fit = right_fit
    else:
        left_fit, right_fit, left_line.radius_of_curvature, right_line.radius_of_curvature, left_fit_x_pos, right_fit_x_pos = sliding_windows_finding_lines(warped, xm_per_pix, ym_per_pix)
        left_line.current_fit = left_fit
        right_line.current_fit = right_fit
    
    if (left_fit_x_pos<=image.shape[1]/2) and (left_fit_x_pos>=0):
        left_line.detected = True
    else:
        left_line.detected = False
        
    if (right_fit_x_pos<=image.shape[1]) and (left_fit_x_pos>=left_fit_x_pos<image.shape[1]/2):
        right_line.detected = True
    else:
        right_line.detected = False
    
    #print(left_line.current_fit)
    #print(right_line.current_fit)
    
    #result = np.dstack((combined_binary, combined_binary, combined_binary))
    #result = hls_filtered
    #result = np.dstack((sobel_binary, sobel_binary, sobel_binary))
    
    lane_drawing = draw_lane(image, left_fit, right_fit, rM)
    result = lane_drawing
        
    return result

def main(__):
    if FLAGS.image is True:
        if FLAGS.image_path is not "":
            if os.path.isfile(FLAGS.image_path) is True:
                filename = FLAGS.image_path.split('/')[-1].split('.')[0]
                #img = cv2.imread(FLAGS.image_path)
                img = mpimg.imread(FLAGS.image_path)
                result = pipeline(img)
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
                ax2.imshow(result)
                cv2.imwrite('./'+filename+'_drawing_lane.jpg', cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            else:
                print("There is no input image...")
        else:
            print("Please add the input image path...")
    elif FLAGS.video is True:
        if FLAGS.video_path is not "":
            if os.path.isfile(FLAGS.video_path) is True:
                filename = FLAGS.video_path.split('/')[-1].split('.')[0]
                clip1 = VideoFileClip(FLAGS.video_path)
                drawing_lane_clip = clip1.fl_image(pipeline)
                drawing_lane_clip.write_videofile('./'+filename+'_drawing_lane.mp4', audio=False)
            else:
                print("There is no input video...")
        else:
            print("Please add the input video path...")
    else:
        print("Please add image or video flag...")

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()


"""
# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]

# Read in an image
img = cv2.imread('test_image.png')

# TODO: Write a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    #undist = np.copy(img)  # Delete this line
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

undistorted = cal_undistort(img, objpoints, imgpoints)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

"""