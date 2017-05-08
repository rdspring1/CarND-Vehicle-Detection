import numpy as np
import scipy.ndimage.measurements
import cv2
from collections import deque

class BoundingBox:
	def __init__(self, img_shape, history_=10, threshold_=0.9, local_threshold_=1):
		img_x, img_y, img_z = img_shape
		self.shape = (img_x, img_y)
		self.threshold = int(history_ * threshold_)
		self.local_threshold = local_threshold_
		self.history = deque(maxlen=history_) 
		self.rnd = 0
		
	def add(self, bbox_list):
		current = np.sum(self.history, axis=0)
        	# Iterate through list of bboxes
		heatmap = np.zeros((self.shape), dtype=np.float)
		for box in bbox_list:
            		# Add += 1 for all pixels inside each bbox
            		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
			heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
		result = np.zeros((self.shape), dtype=np.float)
		result[heatmap >= self.local_threshold] = 1
		result[(heatmap > 0) & (current > 0)] = 1
		self.history.append(result)

	def generate_heatmap(self):
		if len(self.history) == 0:
			return np.zeros(self.shape)
		heatmap = np.sum(self.history, axis=0)
		heatmap[heatmap < self.threshold] = 0
		return heatmap
			
	def draw_labeled_bboxes(self, img):
		heatmap = self.generate_heatmap()
		labels = scipy.ndimage.measurements.label(heatmap)
        	# Iterate through all detected cars
		for car_number in range(1, labels[1]+1):
            		# Find pixels with each car_number label value
			nonzero = (labels[0] == car_number).nonzero()
            		# Identify x and y values of those pixels
			nonzeroy = np.array(nonzero[0])
			nonzerox = np.array(nonzero[1])
            		# Define a bounding box based on min/max x and y
			bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            		# Draw the box on the image
			cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        	# Return the image
		return img 
