import numpy as np
import scipy.ndimage.measurements
import cv2

class BoundingBox:
	def __init__(self, img_shape, threshold_=1, frame_limit_=0):
		img_x, img_y, img_z = img_shape
		self.heatmap = np.zeros((img_x, img_y), dtype=np.float)
		self.threshold = threshold_
		self.frame_limit = frame_limit_
		self.rnd = 0
		
	def add_heat(self, bbox_list):
        # Iterate through list of bboxes
		for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
			self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

		# Zero out pixels below the threshold
		if self.rnd >= self.frame_limit:
			self.heatmap[self.heatmap <= self.threshold] = 0
			self.rnd = 0
		else:
			self.rnd += 1
			
	def draw_labeled_bboxes(self, img):
		labels = scipy.ndimage.measurements.label(self.heatmap)
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
