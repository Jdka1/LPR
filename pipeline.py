from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers import YolosFeatureExtractor, YolosForObjectDetection
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm
import matplotlib.patches as patches
from ultralytics import YOLO
import pytesseract
from orientation.wpodnet.backend import Predictor
from orientation.wpodnet.model import WPODNet
from orientation.wpodnet.stream import ImageStreamer
import errno
import easyocr
from pylab import rcParams
from transformers import YolosForObjectDetection, YolosImageProcessor
from OCR.model import Model as OCR_Model
from OCR.utils import CTCLabelConverter, AttnLabelConverter
import torchvision.transforms as transforms




class LPR_Pipeline():
    def __init__(self, ocr_weights_path):
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.lpd_model = YolosForObjectDetection.from_pretrained('./detection/yolos_model')
        self.feature_extractor = YolosImageProcessor.from_pretrained('./detection/yolos_model')
        
        self.affine_model = WPODNet()
        self.affine_model.load_state_dict(torch.load('weights/wpodnet.pth'))
        self.predictor = Predictor(self.affine_model)
        
        self.yolo_model = YOLO('yolov8m.pt')
        
        # OCR
        self.OCR_config = {
            'weights_path': ocr_weights_path,
            'character': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'imgH': 32,
            'imgW': 100,
            'batch_max_length': 25,
            'rgb': False,
            'sensitive': False,
            'PAD': False,
            'Transformation': 'TPS',
            'FeatureExtraction': 'ResNet',
            'SequenceModeling': 'BiLSTM',
            'Prediction': 'Attn',
            'num_fiducial': 20,
            'input_channel': 1,
            'output_channel': 512,
            'hidden_size': 256
        }
        self.OCR_model, self.OCR_converter = self.load_OCR_model(self.OCR_config)
        self.OCR_model.eval()


        
    def predict_cars(self, image):
        results = self.yolo_model.predict(image, verbose=False)
        boxes = results[0].boxes.xyxy  # xyxy format: (x1, y1, x2, y2)
        confidences = results[0].boxes.conf
        classes = results[0].boxes.cls

        cropped_images = []
        for box, conf, cls in zip(boxes, confidences, classes):
            if conf > 0.9 and int(cls) == 2:  # Car label
                x1, y1, x2, y2 = map(int, box)
                cropped_image_np = image[y1:y2, x1:x2]
                cropped_images.append(cropped_image_np)
        
        return cropped_images
    
    def predict_license_plate(self, image, for_affine=False, confidence_threshold = 0.5):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.lpd_model(**inputs)
        
        
        logits = outputs.logits
        bboxes = outputs.pred_boxes

        probs = torch.softmax(logits, dim=-1)
        confidence_values = probs[..., 1]
        keep = confidence_values > confidence_threshold
        kept_confidence_values = confidence_values[keep]

        image_height, image_width, _ = image.shape
        bboxes = bboxes[keep]
        bboxes[:, [0, 2]] *= image_width
        bboxes[:, [1, 3]] *= image_height
        np_bboxes = bboxes.detach().numpy()

        bbox = np_bboxes[torch.argmax(kept_confidence_values)]
        confidence = torch.max(kept_confidence_values)
        
        center_x, center_y, width, height = bbox            
        if for_affine:
            plate = image[int(center_y-height*1.5):int(center_y+height*1.5), int(center_x-width*1.5):int(center_x+width*1.5)]
        else:
            plate = image[int(center_y-height/2):int(center_y+height/2), int(center_x-width/2):int(center_x+width/2)]
        
        return plate
    
    def predict_affine_plate(self, image):
        prediction = self.predictor.predict(Image.fromarray(image), scaling_ratio=1.0)
        warped = np.array(prediction.warp())
        img = cv2.resize(warped, (128, 64))
        
        return img

    def load_OCR_model(self, model_path):
        converter = AttnLabelConverter(self.OCR_config['character'])
        self.OCR_config['num_class'] = len(converter.character)

        model = OCR_Model(self.OCR_config)
        model = torch.nn.DataParallel(model).to(self.device)
        model.load_state_dict(torch.load(self.OCR_config['weights_path'], map_location=self.device))
        
        return model, converter
    
    def predict_plate_characters(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (self.OCR_config['imgW'], self.OCR_config['imgH']))
        image = resized_image[np.newaxis, :, :].astype(np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0)

        # image = transform(image).unsqueeze(0).to(self.device)

        length_for_pred = torch.IntTensor([self.OCR_config['batch_max_length']]).to(self.device)
        text_for_pred = torch.LongTensor(1, self.OCR_config['batch_max_length'] + 1).fill_(0).to(self.device)

        
        with torch.no_grad():
            preds = self.OCR_model(image, text_for_pred, is_train=False)
            
        # select max probability (greedy decoding) then decode index to character
        preds_size = torch.IntTensor([preds.size(1)] * 1).to(self.device)
        _, preds_index = preds.max(2)
        preds_str = self.OCR_converter.decode(preds_index, preds_size)
        preds_str = preds_str[0].split('[s]')[0].strip().upper()
        
        return preds_str