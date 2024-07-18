import os
import torchvision
import torch


def write_to_txt():
    plates = os.listdir('OCR_fine_tuning/unconstrained_data')
    with open('OCR_fine_tuning/unconstrained_data/_labels.txt', 'w') as f:
        for plate in plates:
            f.write(f'{plate} {plate.split('.')[0]}\n')
    
    
    

