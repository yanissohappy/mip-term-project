import cv2 
import numpy as np
import os
def getLesionMask(img, roi_size): # input 640x480 RGB image

    # green channel
    green_channel = img[:, :, 1]

    # 3 * 3 median filter
    filtered_green_channel = cv2.medianBlur(green_channel, 3) # kernel size == 3, green
    background_img = cv2.medianBlur(green_channel, 25)

    # shaded correction 
    shaded_corrected_img = np.zeros(background_img.shape, np.int) 

    for i in range(480):
        for j in range(640):
            shaded_corrected_img[i][j] = int(filtered_green_channel[i][j]) - int(background_img[i][j])

    shaded_corrected_img = shadedCorrect(shaded_corrected_img)

    kernel_size = 7
    kernel = np.zeros((kernel_size,kernel_size), np.uint8)
    for i in range(kernel_size):
        kernel[kernel_size//2][i] = 1

    # top-hat transformation
    top_hat_img = top_hat(shaded_corrected_img, kernel)

    # open with ellipse kernel to extract the red lesion dots
    filterSize = (3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filterSize) 
    img_open = cv2.morphologyEx(top_hat_img, cv2.MORPH_OPEN, kernel)

    # thresholding
    r, bin_img = cv2.threshold(top_hat_img, 3, 255, cv2.THRESH_BINARY) # testH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)

    # enlarge dots to obtain the mask of ROI
    filterSize = (roi_size, roi_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filterSize) 
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel)

    return bin_img

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def top_hat(img, kernel):
    angles = [i for i in range(0,180,10)]
    img_res = np.zeros(img.shape, np.uint8)
    (height, width) = img_res.shape
    for angle in angles:
        rot_kernel = rotate_image(kernel, angle)
        # print("kernel angle:", angle,"/ 180")
        img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, rot_kernel)

        for i in range(height):
            for j in range(width):
                if img_open[i][j] > img_res[i][j]:
                    img_res[i][j] = img_open[i][j]

    filterSize = (5, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filterSize) 
    img_res = cv2.morphologyEx(img_res, cv2.MORPH_DILATE, kernel)

    for i in range(height):
        for j in range(width):
            if int(img[i][j]) - int(img_res[i][j]) < 0:
                img_res[i][j] = 0
            else:
                img_res[i][j] = img[i][j] - img_res[i][j]
    return img_res

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

def removeFalsePositiveByOpticDisc(img1, img2): # img1 == optic disc mask ; img2 == red lesions candidates
	ret = np.zeros(img1.shape, np.uint8) # 640 * 480
	(height, width) = img1.shape
	for i in range(height):
		for j in range(width):
			if img1[i, j] == 255: # white
				ret[i, j] = 0
			else:
				ret[i, j] = img2[i, j]
	return ret

def getLesionGray(img_gray, lesion_mask):
    lesion_gray = lesion_mask.copy()
    (height, width) = img_gray.shape
    for i in range(height):
        for j in range(width):
            if lesion_mask[i, j] == 255: # white
                lesion_gray[i, j] = img_gray[i, j]
    return lesion_gray
