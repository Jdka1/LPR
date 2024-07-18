import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np

import sys
sys.path.append("../")
from model import create_model
import utils

# Load the model:
model = create_model()
checkpoint = torch.load('model_v2.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()

if torch.cuda.is_available():
  model.to('cuda')

# Prediction pipeline
def pred(image, model):
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  input_tensor = preprocess(image)
  input_batch = input_tensor.unsqueeze(0)

  if torch.cuda.is_available():
      input_batch = input_batch.to('cuda')

  with torch.no_grad():
      output = model(input_batch)['out'][0]
      return output
      
# Loading an image
img = Image.open('plate.jpg').convert('RGB')

# Defining a threshold for predictions
threshold = 0.1 # 0.1 seems appropriate for the pre-trained model

# Predict
output = pred(img, model)

output = (output > threshold).type(torch.IntTensor)
output = output.cpu().numpy()[0]

# Extracting coordinates
result = np.where(output > 0)
coords = list(zip(result[0], result[1]))

# Overlay the original image
for cord in coords:
    img.putpixel((cord[1], cord[0]), (255, 0, 0))
    
img.show()