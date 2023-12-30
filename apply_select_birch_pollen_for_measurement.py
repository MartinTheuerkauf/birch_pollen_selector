# -*- coding: utf-8 -*-


"""

Apply the selection model

"""

import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import glob
import shutil

def classify_flat_betulas(dstf):
    

    model_pth = os.path.join(os.path.dirname(__file__), '..')
    model_name = 'custom-classifier_resnet_18_final_60_last_tr_epochs June23.pth'
    
    # Load the model
    # first, resnet needs to be loaded, and some parameters have to be set..
    model = models.resnet18(pretrained=True)   #load resnet18 model
    num_features = model.fc.in_features #extract fc layers features
    model.fc = nn.Linear(num_features, 2) #(num_of_class == 2)
    model.load_state_dict(torch.load(os.path.join(model_pth, model_name), map_location='cpu'))
    # model.to(device)


    # Define the transformation to apply to the data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   #must same as here
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model.eval()

        
    # create a folder for pollen not suitable for measurements
    pth_unsuited = os.path.join(dstf, 'unsuited')
    if not os.path.isdir(pth_unsuited):
        os.makedirs(pth_unsuited)
    
    imagelist = glob.glob(os.path.join(dstf, '*.tif'))
    # file_name = imagelist[1]
    
    
    for file_name in imagelist:
    
        # Load the image
        print(file_name)
        
        image = Image.open(file_name)
        
    
        # Apply the transformation to the image
        image = transform(image).unsqueeze(0)
        
        # Make a prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            probabilities = torch.nn.functional.softmax(outputs.data, dim=1)
            
        # Print the predicted class and the associated probability
        print('Predicted class:', predicted.item())
        print('Probability:', np.round(probabilities[:, predicted.item()].item(), 4))
        
        file_name = os.path.basename(file_name)
        
        
        # class 0 --> Pollen suitable for measurement, class 1 --> Pollen not suitable for measurement
        # move pollen not suitable for measurement in 'unsuited' 
        if predicted.item() == 1:
            shutil.move(os.path.join(dstf, file_name), pth_unsuited)

        

test_data = os.path.join(os.path.dirname(__file__), '..', 'test_data/')
classify_flat_betulas(test_data)







