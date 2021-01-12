import cv2
import matplotlib.pyplot as plt
import numpy as np
import operator
import scipy.ndimage as ndimage
from skimage.transform import hough_circle, hough_circle_peaks,hough_ellipse
from skimage.draw import circle_perimeter
from skimage import data, color
import os

# red channel	
def findOpticDisc(img): # get rid of artifact
	circled_rgb = img.copy()
	roi_img, roi_leftupper_col, roi_leftupper_row = getROI(img)
	b, g, red_channel = cv2.split(roi_img)
	red_channel_blurred_img = cv2.GaussianBlur(red_channel, (5, 5), 0)
	red_channel = cv2.addWeighted(red_channel, 1.5, red_channel_blurred_img, -0.5, 0, red_channel)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
	red_channel = ndimage.grey_closing(red_channel, structure = kernel)
	red_channel = cv2.equalizeHist(red_channel)

	edged_img = canny(red_channel, 0.22)
	rect_kernel = np.ones((3, 3), np.uint8)
	edged_img = cv2.dilate(edged_img, rect_kernel, iterations = 3)
	accums, c_col, c_row, radii = hough(edged_img, 55, 80)
	for center_row, center_col, radius in zip(c_row, c_col, radii):
		circ_row, circ_col = circle_perimeter(center_row, center_col, radius)
		try:
			roi_img[circ_row, circ_col] = (100, 180, 120)
		except :
			continue	

	center_row = int((roi_leftupper_row + center_row) * 0.6667)
	center_col = int((roi_leftupper_col + center_col) * 0.625)

	cv2.circle(circled_rgb, (center_col, center_row), 40, (0, 255, 0), 1)

	mask_img = np.zeros(img.shape[:2], np.uint8)
	cv2.circle(mask_img, (center_col, center_row), 40, (255, 255, 255), -1)

	return circled_rgb, mask_img

# shadedCorrect function 
def shadedCorrect(v):
	ret = np.zeros(v.shape, np.uint8) 
	for i in range(len(v)):
		for j in range(len(v[0])):
			if v[i][j] >= 0: 
				ret[i][j] = 0
			else: 
				ret[i][j] = abs(v[i][j])
	return ret

def getROI(img): # ROI == region of interest
	img = cv2.resize(img, (1024, 720), interpolation = cv2.INTER_CUBIC)
	b, g, r = cv2.split(img)
	g = cv2.GaussianBlur(g, (15,15), 0)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
	g = ndimage.grey_opening(g, structure = kernel)	
	(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(g)

	x0 = int(maxLoc[0]) - 110
	y0 = int(maxLoc[1]) - 110
	x1 = int(maxLoc[0]) + 110
	y1 = int(maxLoc[1]) + 110
	center_row = maxLoc[0]
	center_col = maxLoc[1]
	
	return img[y0:y1, x0:x1], x0, y0
	
def canny(img, sigma):
	v = np.mean(img)
	sigma = sigma
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged_img = cv2.Canny(img, lower, upper)	
	return edged_img
	
def hough(edged, limm, limM):
	hough_radii = np.arange(limm, limM, 1)
	hough_res = hough_circle(edged, hough_radii)
	return hough_circle_peaks(hough_res, hough_radii, total_num_peaks = 1)