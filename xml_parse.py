import xml.etree.ElementTree as ET
import os
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import cv2

def getGroundTruth(img_name): # return 1 if image contains red lesion, 0 otherwise
    file_group = getFileGroup(img_name)
    expert_think_it_has_red_lesion = 0
    for a_file in file_group: # 01~04
        #print(a_file)
        check = False
        # parse xml	
        tree = ET.parse(a_file)
        root = tree.getroot()
        
        for marking in root.findall("./markinglist/marking"): # label "marking" is under label "markinglist"
            for markingtype in marking.findall("./markingtype"): # get its pathology type
                if markingtype.text.strip() == "Haemorrhages" or markingtype.text.strip() == "Red_small_dots": # red lesions
                    if check == False:
                        expert_think_it_has_red_lesion += 1
                        check = True
                    # for coords2d in marking.iter('coords2d'): # iteratively find (only in every label "marking")
                        # print("coords2d:", coords2d.text)
                # print("------")

    if expert_think_it_has_red_lesion == 4:
        return 1
    else:
        return 0

# get file group
def getFileGroup(img_name): # e.g. image001.png
    num_annotations = 4
    filenames = []
    for i in range(1,num_annotations+1):
        filenames.append("ddb1_v02_01/groundtruth/diaretdb1_"+img_name[:8]+"_0"+str(i)+"_plain.xml")
        
    return filenames

############################
# 20201223~20201225 update #
############################
# All the locations of the following functions are (row, col)
def getLesionLocationFromLesionCandidatesMask(mask_img): # input: NewLesionCandidates; output: location of circle (x, y)
	_, mask_img = cv2.threshold(mask_img,0,255,cv2.THRESH_OTSU)
	h,w = mask_img.shape[:2]

	contours0, hierarchy = cv2.findContours(mask_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	moments  = [cv2.moments(cnt) for cnt in contours0]
	centroids = [( int(round(m['m01']/m['m00'])), int(round(m['m10']/m['m00']))) for m in moments if m['m00'] != 0 and m['m00'] != 0] # list 
	
	return sorted(centroids, key = lambda element: (element[0], element[1]))
	
def getLesionLocationFromXML(xml_file): # input: "./diaretdb1_image001_01_plain.xml"(1500*1152); output: location of lesion (x, y) for size 640*480 image
	tree = ET.parse(xml_file)
	root = tree.getroot()
	coordinates = []

	for marking in root.findall("./markinglist/marking"): 
		for markingtype in marking.findall("./markingtype"): 
			if markingtype.text.strip() == "Haemorrhages" or markingtype.text.strip() == "Red_small_dots": # red lesions
				for representativepoint in marking.iter('representativepoint'):
					for coords2d in representativepoint.findall("./coords2d"): 
						# print("representativepoint:", coords2d.text)			
						# I use row and col as coordinates
						center_col = int(int(coords2d.text.split(",")[0]) * 0.426667) # y
						center_row = int(int(coords2d.text.split(",")[1]) * 0.416667) # x
						coordinates.append([center_row, center_col])
	return sorted(coordinates, key = lambda element: (element[0], element[1]))

# for GLCM use
def makeRoiImage(img, img_name, coordinates, range, write_path): # img: original image; img_name: "image001.png"; coordinates: list of list; range: ROI side length
	(height, width) = img.shape[:2]
	for coordinate in coordinates:
		(center_row, center_col) = (coordinate[0], coordinate[1])
		top = center_row - range
		bottom = center_row + range
		left = center_col - range
		right = center_col + range
		if top < 0:
			top = 0
		if bottom >= height:
			bottom = height - 1
		if left < 0:
			left = 0
		if right >= width:
			right = width - 1
		# fool proof XD
		if top >= bottom: 
			top = bottom
		if left >= right:
			left = right
		file_path = write_path + '/' + img_name[:8] + '_' + str(center_row) + "_" + str(center_col) + ".png"
		cv2.imwrite(file_path , img[top: bottom + 1, left: right + 1])
		
def writeCoordinateInFile(xml_file, mask_img, write_path):
	txt_file = open(write_path,"w") 

	coordinates_xml = getLesionLocationFromXML(xml_file)
	coordinates_mask = getLesionLocationFromLesionCandidatesMask(mask_img)
	
	for coordinate in coordinates_xml:
		txt_file.write(str(coordinate[0])+ ' ' +str(coordinate[1]))
		txt_file.write('\n')
	txt_file.write('total number of XML: ' + str(len(coordinates_xml)))
	txt_file.write('\n-----------\n')
	for coordinate in coordinates_mask:
		txt_file.write(str(coordinate[0])+ ' ' +str(coordinate[1]))
		txt_file.write('\n')
	txt_file.write('total number of our mask: ' + str(len(coordinates_mask)))
	txt_file.write('\n-----------\n')
	txt_file.close()
	
	findClosePair(coordinates_xml, coordinates_mask, write_path, 20)
	
	
def findClosePair(coordinates_xml, coordinates_mask, write_path, discard_dist): # discard_dist: if dist is over the range, get rid of it
	txt_file = open(write_path,"a")
	count = 0
	for coordinate1 in coordinates_xml:
		# xml_row = coordinate1[0]
		# xml_col = coordinate1[1]
		row_min = 0
		col_min = 0
		dist_min = 99999
		for coordinate2 in coordinates_mask:
			# mask_row = coordinate2[0]
			# mask_col = coordinate2[1]			
			dist = abs(np.linalg.norm(np.array(coordinate1) - np.array(coordinate2))) # get distance
			if dist <= dist_min:
				row_min = coordinate2[0]
				col_min = coordinate2[1]
				dist_min = dist
		if dist_min <= discard_dist:
			txt_file.write(str(coordinate1[0]) + ' ' + str(coordinate1[1]) + ' | ' + str(row_min)+ ' ' +str(col_min))
			txt_file.write('\n')
			count += 1
	txt_file.write('total number of close pair: ' + str(count))
	txt_file.close()

############################
# 20201226 update 		   #
############################

# coordinate_mask: only one coordinate
# discard_dist: if dist is over the range, get rid of it
def existClosePair(coordinate_mask, coordinates_xml, discard_dist):
	for coordinate_xml in coordinates_xml:
		dist = abs(np.linalg.norm(np.array(coordinate_xml) - np.array(coordinate_mask))) # get distance
		if dist <= discard_dist:
			return True
	return False

def createFeaturesAndGroundTruthFile_v2(lesion_mask, img_gray, img_name, features_write_path):
	features_file = open(features_write_path,"w")

	mask_coordinates = getLesionLocationFromLesionCandidatesMask(lesion_mask)
	roi_images = getRoiImages(img_gray, mask_coordinates, 18) # range; main_v2: 4; main_v4: 20, 18

	file_group = getFileGroup(img_name)
	xml_coordinates_list = []
	for xml_file in file_group:
		xml_coordinates_list.append(getLesionLocationFromXML(xml_file))
	
	for i in range(len(mask_coordinates)):
		mask_coordinate = mask_coordinates[i]
		# write ground truth to file
		exist_pair_cnt = 0
		for xml_coordinates in xml_coordinates_list:
			exist_pair = existClosePair(mask_coordinate, xml_coordinates, 20) # distance; main_v2: 18; ; main_v4: 10, 18
			if exist_pair:
				exist_pair_cnt += 1
		if exist_pair_cnt == 4:
		# if exist_pair_cnt >= 1:
			features_file.write('1 ')
		else:
			features_file.write('0 ')

		# Get GLCM features
		# angles = [0, np.pi/4, np.pi/2, np.pi*3/4]
		angles = [0, np.pi/8, np.pi/4, np.pi*3/8, np.pi/2, np.pi*5/8, np.pi*3/4, np.pi*7/8]
		glcm = greycomatrix(roi_images[i], [1], angles)
		# filt_glcm = glcm[1:, 1:, :, :]
		filt_glcm = glcm

		contrast = greycoprops(filt_glcm, prop='contrast')
		for i in range(len(angles)):
			features_file.write(str(contrast[0][i])+' ')
		# features_file.write('\n')

		dissimilarity = greycoprops(filt_glcm, prop='dissimilarity')
		for i in range(len(angles)):
			features_file.write(str(dissimilarity[0][i])+' ')
		# features_file.write('\n')

		homogeneity = greycoprops(filt_glcm, prop='homogeneity')
		for i in range(len(angles)):
			features_file.write(str(homogeneity[0][i])+' ')
		# features_file.write('\n')

		ASM = greycoprops(filt_glcm, prop='ASM')
		for i in range(len(angles)):
			features_file.write(str(ASM[0][i])+' ')
		# features_file.write('\n')

		energy = greycoprops(filt_glcm, prop='energy')
		for i in range(len(angles)):
			features_file.write(str(energy[0][i])+' ')
		# features_file.write('\n')

		correlation = greycoprops(filt_glcm, prop='correlation')
		for i in range(len(angles)):
			features_file.write(str(correlation[0][i])+' ')
		features_file.write('\n')

		# features = [contrast, dissimilarity, homogeneity, ASM, energy, correlation]

	features_file.close()

def getRoiImages(img_gray, coordinates, range): # img: original image; img_name: "image001.png"; coordinates: list of list; range: ROI side length
	(height, width) = img_gray.shape[:2]
	roi_images = []
	for coordinate in coordinates:
		(center_row, center_col) = (coordinate[0], coordinate[1])
		top = center_row - range
		bottom = center_row + range
		left = center_col - range
		right = center_col + range
		if top < 0:
			top = 0
		if bottom >= height:
			bottom = height - 1
		if left < 0:
			left = 0
		if right >= width:
			right = width - 1
		# fool proof XD
		if top >= bottom: 
			top = bottom
		if left >= right:
			left = right

		roi_images.append(img_gray[top: bottom + 1, left: right + 1])
	return roi_images