# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image0]: ./output_images/car.png
[image1]: ./output_images/noncar.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/all_sliding_windows.png
[image4]: ./output_images/positive_sliding_window.png
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view)

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image0]
![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for the `skimage.hog()` output.

Here is an example using the `YCrCb` color space and HOG parameters of 'orientations=9', 'pixels_per_cell=(8, 8)', and 'cells_per_block=(2, 2)':

![alt text][image2]


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the sklearn toolkit. The code is found in the fifth code cell in the IPython notebook.
The data is preprocessed to have zero mean and unit variance. I attempted to use a non-linear SVM to improve performance and generalization using the 'RBF' kernel trick. I achieve between 98.5-99% accuracy with the linear SVM, but with the non-linear SVM, my test accuracy was 99%. However, the non-linear SVM was significantly slower, so I did not use it with my pipeline. I suspect it would have fewer false positives while processing the video. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The 'search_window' function is found in feature_extraction.py. I search the lower quarter section of the image between (400, 700). I extract HOG features only once for the entire region of interest in each full image. I then subsample windows from the region of interest.

I chose to move 2 cells_per_step to avoid false positives. When I tried 1 cell_per_step, I could detect cars better in a few frames, but saw a greater increase in false positives.

I tried an alterative scaling factor - (1.0 x-axis and 1.5 y-axis). I detected more bounding-boxes per car, but it did not improve overall accuracy. Since decreasing the scaling factor can slow down the pipeline, I went with the default scaling factor of 1.5 for the x and y axes.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.
What did you do to optimize the performance of your classifier?

Ultimately, I searched used all three YCrCb color channels to gather HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  

Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here is [my video](./output_images/result_video.mp4).


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

From the set of positive detections, I sum them together to create a heatmap. I apply a threhold to this heatmap to avoid false positive detections. Then, I add this heatmap to a deque that tracks a series of heatmap frames. This deque creates a moving average of the 30 video frames. To identify the vehicle positions, I take the mean of the images in the deque. I also threshold this mean image to ignore detections that did not appear in at least 40% of the deque's frames. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I constructed bounding boxes to cover the area of each blob detected.

The first threshold avoids obvious false positives. I uses a large frame buffer to track the car better when the pipeline misses the car in the frame. The second threshold avoids any false positives the buffer may accumulate over time. This setup works well at avoiding false postives and false negatives in the video.  

### Here are six frames and their corresponding heatmaps:
![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

+ False Negatives - My pipeline still loses a car for a couple of frames when it is obviously visible. I tried to lower my filter's threshold, but this lead to more false positives. I decide that it was better to tradeoff a few missing frames to avoid any false positives.

+ Bounding Box Lag - My pipeline uses a large frame buffer to track the cars smoothly. However, the bounding box misses the front of the car and captures the open space just behind the car. The centroid of the car is cover well, but it is not a perfect match.

Perhaps, a Kalman Filter can track the position and velocity of the car, which could predict the next position of the car. It would solve both problems of False Negatives and the lag in the bounding box.
