import optic_disc as od
import lesion_candidate as lc
from knn import knn
from xml_parse import createFeaturesAndGroundTruthFile_v2, getGroundTruth

import os
import cv2 
import numpy as np

folder_path = './ddb1_v02_01/images'
all_img_list = os.listdir(folder_path)
lesion_cnt = 0
non_lesion_cnt = 0
for img_name in all_img_list:	
    print(folder_path + '/' + img_name)
    img = cv2.imread(folder_path + '/'  + img_name, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (640,480), interpolation = cv2.INTER_AREA)

    # Get optic mask
    optic_img, optic_mask = od.findOpticDisc(img)
    optic_img_write_path = 'Results/optic_img/optic_' + img_name
    optic_mask_write_path = 'Results/optic_mask/optic_mask_' + img_name
    cv2.imwrite(optic_img_write_path, optic_img)
    cv2.imwrite(optic_mask_write_path, optic_mask)	

    # Get lesion candidates
    lesion_img = lc.getLesionMask(img, 5)
    lesion_img_write_path = 'Results/LesionCandidates/lesion_' + img_name
    cv2.imwrite(lesion_img_write_path, lesion_img)

    # Remove false positive in optic disc region
    removed_false_positive_candidate_img = lc.removeFalsePositiveByOpticDisc(optic_mask, lesion_img)
    lesion_img_write_path = 'Results/NewLesionCandidates/lesion_' + img_name
    cv2.imwrite(lesion_img_write_path, removed_false_positive_candidate_img)

    # Get the gray image of the lesion candidate regions
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Write Features and Ground Truth to file
    features_write_path = 'Results/FeaturesAndLabel/' + img_name[:8] + '.txt'
    createFeaturesAndGroundTruthFile_v2(removed_false_positive_candidate_img, img_gray, img_name, features_write_path)

# Machine Learning
f = open('trainset.txt', 'r')
X_train = []
for img_name in f.readlines():
    data_file = open('Results/FeaturesAndLabel/' + img_name[:8] + '.txt', 'r')
    pred_label = 0
    for line in data_file.readlines():
        data = line.split()
        x_line = []
        for i in range(len(data)):
            x_line.append(float(data[i]))
        X_train.append(x_line)

f = open('testset.txt', 'r')
true_labels = []
predicted_labels = []
lesion_labels1 = []
for img_name in f.readlines():
    true_label = getGroundTruth(img_name)
    true_labels.append(true_label)

    # check each lesion in the image
    data_file = open('Results/FeaturesAndLabel/' + img_name[:8] + '.txt', 'r')
    label = 0

    for line in data_file.readlines():
        data = line.split()
        x_line = []
        for i in range(len(data)):
            x_line.append(float(data[i]))

        k = 3
        ret, label = knn(X_train, x_line[1:], k)
        if label == 1:
            break

    predicted_labels.append(label)
        
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
for i in range(len(true_labels)):
    if predicted_labels[i] == 0:
        if true_labels[i] == 0:
            true_negative += 1
        else:
            false_negative += 1
    else:
        if true_labels[i] == 0:
            false_positive += 1
        else:
            true_positive += 1

sensitivity = true_positive / (true_positive+false_negative)  
specificity = true_negative / (true_negative+false_positive)
print('True positive =', true_positive)
print('False negative =', false_negative)
print('True negative =', true_negative)
print('False positive =', false_positive)
print('True positive + False Negative =', true_positive+false_negative)
print('True negative + false positive =', true_negative+false_positive)
print('Sensitivity =', sensitivity)
print('Specificity =', specificity)