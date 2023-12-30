# -*- coding: utf-8 -*-

"""
code inspired by
https://medium.com/nerd-for-tech/image-classification-using-transfer-learning-pytorch-resnet18-32b642148cbe
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object


train_dir = os.path.join(os.path.dirname(__file__), '..', 'data/trainfol/')
train_classa_dir = os.path.join(os.path.dirname(__file__), '..', 'data/trainfol/Betula_flat_train/')
train_classb_dir = os.path.join(os.path.dirname(__file__), '..', 'data/trainfol/Betula_nonflat_train/')

test_dir = os.path.join(os.path.dirname(__file__), '..', 'data/testfol/')
test_classa_dir = os.path.join(os.path.dirname(__file__), '..', 'data/testfol/Betula_flat_test/')
test_classb_dir = os.path.join(os.path.dirname(__file__), '..', 'data/testfol/Betula_nonflat_test/')


# --------------preprocessing-------------------------------------------------------------------------

transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(), # data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
])

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
     transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# We need to load the dataset with torchvision.dataset.ImageFolder function by applying to preprocess.
train_dataset = datasets.ImageFolder(train_dir, transforms_train)
test_dataset = datasets.ImageFolder(test_dir, transforms_test)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=0)


# You can check the training/testing images count and their class names with the below code. (Not necessary Step)
print('Train dataset size:', len(train_dataset))
print('Test dataset size:', len(test_dataset))
class_names = train_dataset.classes
print('Class names:', class_names)


# You can visualize random images below code from the training dataset (Not a necessary Step)
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 60
plt.rcParams.update({'font.size': 20})

def imshow(input, title):
    # torch.Tensor => numpy
    input = input.numpy().transpose((1, 2, 0))
    # undo image normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # display images
    plt.imshow(input)
    plt.title(title)
    plt.show()

# load a batch of train image
iterator = iter(train_dataloader)

# visualize a batch of train image
inputs, classes = next(iterator)
out = torchvision.utils.make_grid(inputs[:4])
imshow(out, title=[class_names[x] for x in classes[:4]])


# ------------------------Training Steps:---------------------------------------------------------------------
# We need to download resnet18 pre-trained weights, and change its layers 
# because we need to classify specific classes, while Resnet-18 is trained on many classes. 
# You can use any optimizer and loss function, I have used SGD optimizer and Cross-Entropy loss. 
# You can use the below code to download the Resnet-18 model and tune its layers.

model = models.resnet18(pretrained=True)   #load resnet18 model
num_features = model.fc.in_features     #extract fc layers features
model.fc = nn.Linear(num_features, 2) #(num_of_class == 2)
model = model.to(device) 
criterion = nn.CrossEntropyLoss()  #(set loss function)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# Now, we need to start training, if the above steps work fine, 
# then you can easily start training with the below code.

num_epochs = 60   #(set no of epochs)
start_time = time.time() #(for showing time)
for epoch in range(num_epochs): #(loop for every epoch)
    
    print("Epoch {} running".format(epoch)) #(printing message)
    """ Training Phase """
    model.train()    #(training model)
    running_loss = 0.   #(set loss 0)
    running_corrects = 0 
    
    # load a batch data of images
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device) 
        
        # forward inputs and get output
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        # get loss value and update the network weights
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset) * 100.
    print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() -start_time))
    
    """ Testing Phase """
    model.eval()
    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = running_corrects / len(test_dataset) * 100.
        print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time()- start_time))

# Now, we need to save our trained model for testing and further usage. 
# You can use the below code for saving the model in the file.

save_path = 'test_model.pth'
torch.save(model.state_dict(), save_path)







#------------------- Testing Steps---------------------------------------------------------------------------------------------

# Step-12: For testing, we first need to load over the trained model. 
# you can use the below code for the loading model.

model = models.resnet18(pretrained=True)   #load resnet18 model
num_features = model.fc.in_features #extract fc layers features
model.fc = nn.Linear(num_features, 2)#(num_of_class == 2)
model.load_state_dict(torch.load("Acne-classifier_resnet_18_final_60_last_tr_epochs.pth"))
model.to(device)

# Now, we can test our model, on testing data.

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 60
plt.rcParams.update({'font.size': 20})
def imshow(input, title):
    
    # torch.Tensor => numpy
    input = input.numpy().transpose((1, 2, 0))
    
    # undo image normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    
    # display images
    plt.imshow(input)
    plt.title(title)
    plt.show()##Testing

model.eval()
start_time = time.time()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
with torch.no_grad():
    running_loss = 0.
    running_corrects = 0
    for i, (inputs, labels) in enumerate(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        if i == 0:
            print('======>RESULTS<======')
            images = torchvision.utils.make_grid(inputs[:4])
            imshow(images.cpu(), title=[class_names[x] for x in labels[:4]])
    epoch_loss = running_loss / len(test_dataset)
    epoch_acc = running_corrects / len(test_dataset) * 100.
    print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.
          format(epoch, epoch_loss, epoch_acc, time.time() - start_time))






"""
Apply the model (nach chatGPT)

"""

import torch
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import torch.nn as nn
# import numpy as np
from PIL import Image
import os
import glob

os.chdir(r'D:\Arbeit\Lüneburg\Paper 1 Betula\training_recognition\data_Betula100')

# Load the model
# first, resnet needs to be loaded, and some parameters have to be set..
model = models.resnet18(pretrained=True)   #load resnet18 model
num_features = model.fc.in_features #extract fc layers features
model.fc = nn.Linear(num_features, 2) #(num_of_class == 2)
model.load_state_dict(torch.load('custom-classifier_resnet_18_final_60_last_tr_epochs.pth', map_location='cpu'))


# Define the transformation to apply to the data
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


srcf = r'D:\Arbeit\Lüneburg\Paper 1 Betula\KM23B 500-600\KM23B 500-600 94-95'
imagelist = glob.glob(os.path.join(srcf, '*.tif'))

for file_name in imagelist:

    # Load the image
    image = Image.open(file_name)
    
    # Apply the transformation to the image
    image = transform(image).unsqueeze(0)
    
    # Make a prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        probabilities = torch.nn.functional.softmax(outputs.data, dim=1)
        
    # Print the predicted class and the associated probability
    # print('Predicted class:', predicted.item())
    # print('Probability:', np.round(probabilities[:, predicted.item()].item(), 4))
    
    print(predicted.item(), np.round(probabilities[:, predicted.item()].item(), 4))

        









