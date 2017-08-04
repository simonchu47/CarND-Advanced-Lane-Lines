
##Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted.jpg "Undistorted"
[image2]: ./output_images/undistorted_test5.jpg "Road Transformed"
[image3]: ./output_images/test5_filtered.jpg "Filtered Image"
[image4]: ./output_images/dynamic_filter.jpg "Filtered Image"
[image5]: ./output_images/perspective_transformed.jpg "Warp Example"
[image6]: ./output_images/finding_pixels.jpg "Fit Visual"
[image7]: ./output_images/test5_drawing_lane "Output"
[video1]: ./output_images/project_video_drawing_lane_normal "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how I addressed each one.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how I computed the camera matrix and distortion coefficients.

The code for this step is contained in the in lines 555 through 623 of the file called `advancedFindingLane.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a union of two HLS color space filters and one R channel gradient filter to generate a binary image (thresholding steps at lines 666 through 684 in `advancedFindingLane.py`). One HLS filter is to find out the pixels which represents the yellow lines, and the other one is for white lines. The R channel gradient filter is to find out more white lines' edges.

'Here's an example of the filterd image for this step. 

![alt text][image3]

In addition, a dynamic filter concept was also designed in. When the funtion is enabled via command line as `--dynamic_filter`, on each frame, the lower threshold of L channel for white line HLS filter is decided according to the average of L channel value of the road surface for the last 5 iterations. That is to increase the robustness of the lane detection.

'Here's an example of the filtered image with dynamic filter function enabled. From the challenge video, beneath the bridge, if the original fixed threshold filter is used, there is no pixel of road surface part left on the image. But if the dynamic threshold filter is used, the pixels representing white lines are left.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes the functions from Project 1, which appears in lines 119 through 291 in the file `advancedFindingLane.py`. The `gen_perspective_trans_matrix()` function takes as inputs an image (`img`) and automatically generates source (`src`) and destination (`dst`) points.  The method is from Project 1 implementation, which finds out two straight lines which represent the left and right edge of the lane respectively. The start and end points of both lines form the four corners of a trapezoid which should become rectangular after perspective transformation.

I verified that my perspective transform was working as expected by apply the `gen_perspective_trans_matrix()`function onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then there are two methods implemented to find out lane-line pixels and fit the lane lines with a 2nd order polynomial. The first one is sliding window method, which needs more calculation power. But if both the left and right line-fitting are known, the second method would be used, which is more efficient.

And after the pixels are found, `np.polyfit()`is called to get the second order fitting polynomial. All the functions appear in lines 332 through 480 in the file `advancedFindingLane.py`. The following shows the results with sliding window searching method and known fitting searching method respectively.

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The calculation of left and right lane-line curvature is implemented in line 483 through 485 as `calculate_curve_rad()` function which uses the newly re-fitted polynomial that was calculated by function 'fit_scaled_polynomial()', in line 493 through 497, which re-fits the pixels that are generated from a known line-fitting and scaled in the x and y direction by different scalers repectively. These steps are in the pipeline, from line 781 through 784, namely, `left_fit_cr` and `right_fit_cr` which are new line-fitting for pixels in meter unit, and both the curve radius which are calcualted based on `left_fit_cr` and `right_fit_cr` and are in meter unit as well.

The vehicl center is calculated in line 785, on which the function `calculate_vehicle_position()` is called. That function is implemented in line 499 through 503, and is only related to x position. Therefore the calculation is based on the x-coordinates of pixels and scaled in the x direction.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 505 through 530 in my code in `advancedFindingLane.py` in the function `draw_lane()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [video1](./project_video_drawing_lane_normal.mp4)
(The video path is `./project_video_drawing_lane_normal.mp4`)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. The automatic generation of perspective transformation matrix is implemented. Without hard code I used the hough line methods to find out the four corners for before and after the perspective transformation. It's pretty convenient, but of course, you have to make sure that the picture you used must have a straight lane in front of you.

2. The union of multiples of HLS color space filters are used. Each filter is designed for specific function. For example, one for finding out yellow lines and another one for white lines. It's almost impossible to find out both yellow and white lines through one set of HLS thresholds.

3. The R channel gradients filter is added into the union of the filters. The R-ch gradients filter is to help the white-lines filter to find out more edges of white lines. It also could help yellow-lines filter, but it might cause some side effects that leave unnecessary pixels when noises appear on the image, such as shadows or taints on the road surface.

4. The dynamic filter concept is designed in. The implementation is for the white-line filter, for the reason that the lower threshold of L channel is sensible to the environments. The setting for project video is not suitable for challenge video. Even on the same video, there might be different environments, such as beneath a bridge. But the method has to run with a good algorithm for how to modify the parameters. Currently I just added 30 to the measured L ch. value of road surface for the lower threshold. That's simple but creates limited improvements. I would develop more advanced algorithm for more other filters to gain more robustness.
The following is a demo with or without dynamic filter function enabled, and you can see the improvements.

    Without dynamic filter [video2](./challenge_video_drawing_lane_normal.mp4)
    (The video path is `./challenge_video_drawing_lane_normal.mp4`)

    With dynamic filter [video3](./examples/challenge_video_drawing_lane_dy.mp4)
    (The video path is `./challenge_video_drawing_lane_normal_dy.mp4`)

5. The moving average is used on the line-fitting polynomials' coefficients for last several iterations. On each frame the program records the averaged polynomial on which the searching for line pixels is based. The averaged polynomial is also used for the lane drawing and calculation of vehicle center. It works like a low pass filter and prevents jitters.
The following is the same video as above but with the moving average function enabled.

    With moving average function [video4](./examples/challenge_video_drawing_lane_ma.mp4)
    (The video path is `./challenge_video_drawing_lane_normal_ma.mp4`)

6. If both dynamic filter and moving average methods are used, the performance would be more stable, like the following demo.
    
    [video5](./examples/challenge_video_drawing_lane_ma_dy.mp4)
    (The video path is `./challenge_video_drawing_lane_normal_ma_dy.mp4`)


7. However, my pipeline with dynamic filter enabled will sometimes let the fitting lines wobble. The possible reason is that currently I just added 30 to the measured L ch. value of road surface as the lower threshold of L ch. for white-line filter, and the added value is not enough for some kind of road surface. The solution might be the development of an advanced algorithm for the tuning of thresholds values, or with a look-up-table.

8. Tested with the harder challenge video, my pipeline would still fail when glare or reflection appeared on the image. The solution would be that a more robust combination of filters should be designed, and it would be used when the glare and reflection situation is detected by the method of dynamic filter turned on.









