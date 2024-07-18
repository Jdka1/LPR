import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_utils import read_image_and_text
from pipeline import LPR_Pipeline






# Example usage
directory = 'OCR_fine_tuning/us/'
image_text_pairs = read_image_and_text(directory)
image_text_pairs = list(map(lambda pair: (pair[0], pair[1].split('\n')[0].split('\t')), image_text_pairs))



predictions = []
real_values = []
filenames = list(map(lambda pair: pair[1][0], image_text_pairs))
for i in tqdm(range(len(image_text_pairs))):
    try:
        # car = lpr.predict_cars(image_text_pairs[i][0])[0]
        plate = lpr.predict_license_plate(image_text_pairs[i][0], for_affine=True)
        plate = lpr.predict_affine_plate(plate)
        
        chars = lpr.predict_plate_characters(plate)

        predictions.append(chars)
        real_values.append(image_text_pairs[i][1][-1])
        
        write_path = 'OCR_fine_tuning/unconstrained_data'
        cv2.imwrite(f'{write_path}/plate{i}.jpg', plate)
    except Exception as e:
        continue
    
    
    


directory = 'OCR_fine_tuning/us/'
image_text_pairs = read_image_and_text(directory)
image_text_pairs = list(map(lambda pair: (pair[0], pair[1].split('\n')[0].split('\t')), image_text_pairs))


lpr = LPR_Pipeline(
    ocr_weights_path='OCR_fine_tuning/saved_models/train_6/best_accuracy.pth'
)

predictions = []
real_values = []
for i in tqdm(range(len(image_text_pairs))):
    try:
        # car = lpr.predict_cars(image_text_pairs[i][0])[0]
        plate = lpr.predict_license_plate(image_text_pairs[i][0], for_affine=True)
        plate = lpr.predict_affine_plate(plate)
        
        chars = lpr.predict_plate_characters(plate)

        predictions.append(chars)
        real_values.append(image_text_pairs[i][1][-1])
        write_path = 'OCR_fine_tuning/unconstrained_data'
        cv2.imwrite(f'{write_path}/{image_text_pairs[i][1][-1]}.jpg', plate)
    except Exception as e:
        print(image_text_pairs[i][1][0])
        continue