import numpy as np
import cv2
from skimage.feature import hog

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
	
def extract_features(directory, imgs, color_space='BGR', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors
	features = []
    # Iterate through the list of images
	for file in imgs:
		file_features = []
		img = cv2.imread(directory + file)
		img_features = single_img_features(img, spatial_size,
                        hist_bins, orient, 
                        pix_per_cell, cell_per_block, hog_channel)
		features.append(img_features)
	return features

# Define a function to extract features from an image
def single_img_features(img, spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):					
    #1) Define an empty list to receive features
    img_features = []
    norm_img = img.astype(np.float32)/255.0
	
	#2) Apply color conversion
    feature_image = cv2.cvtColor(norm_img, cv2.COLOR_BGR2YCrCb)
			
    #3) Compute spatial features if flag is set
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    img_features.append(spatial_features)
	
    #5) Compute histogram features if flag is set
    hist_features = color_hist(feature_image, nbins=hist_bins)
    img_features.append(hist_features)
	
    #7) Compute HOG features if flag is set
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.extend(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))      
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
    img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

def search_windows(img, ystart, ystop, scale, svc, scaler, 
                    spatial_size=(32, 32), hist_bins=32, 
                    orient=9, pix_per_cell=8, cell_per_block=2):
	# normalize
	norm_img = img.astype(np.float32)/255.0
	
	# crop search area
	search_image = norm_img[ystart:ystop,:,:]
	
	# convert color space
	search_image = cv2.cvtColor(search_image, cv2.COLOR_BGR2YCrCb)
	
	if scale != 1.0:
		imshape = search_image.shape
		search_image = cv2.resize(search_image, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
		
	ch1 = search_image[:,:,0]
	ch2 = search_image[:,:,1]
	ch3 = search_image[:,:,2]

    # Define blocks and steps as above
	nxblocks = (ch1.shape[1] // pix_per_cell)-1
	nyblocks = (ch1.shape[0] // pix_per_cell)-1
	nfeat_per_block = orient*cell_per_block**2
	
	window = 64
	nblocks_per_window = (window // pix_per_cell)-1
	cells_per_step = 2  # Instead of overlap, define how many cells to step
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
	hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
	hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
	hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
	
	on_windows = []
	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step
			
			hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
			hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
			hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
			
			xleft = xpos * pix_per_cell
			ytop = ypos * pix_per_cell

            # Extract the image patch
			subimg = cv2.resize(search_image[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
			spatial_features = bin_spatial(subimg, size=spatial_size)
			hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
			test_features = scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
			
			#6) Predict using your classifier
			prediction = svc.predict(test_features)
			
			#7) If positive then save the window
			if prediction == 1:
				xbox_left = np.int(xleft*scale)
				ytop_draw = np.int(ytop*scale)
				win_draw = np.int(window*scale)
				on_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    #8) Return windows for positive detections
	return on_windows