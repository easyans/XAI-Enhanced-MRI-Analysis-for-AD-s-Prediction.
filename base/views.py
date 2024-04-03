from django.shortcuts import render,HttpResponse,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required
# Create your views here.

from django.shortcuts import render
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
global scaler
from django.conf import settings
import os

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
###
#import the required libraries
import pathlib
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import PIL

from django.shortcuts import redirect
####
from PIL import Image
import torchvision.transforms as transforms
import torch
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.shortcuts import render
import os
import numpy as np

import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn.functional as F
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.conf import settings
from PIL import Image
import torch
from torchvision import transforms
import os
from django.http import HttpResponseBadRequest
######################
# Import necessary libraries
from django.http import HttpResponse
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor
import torch

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Define your UNet architecture here
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Implement the forward pass
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('alzeimer_alexnet_model.pth', map_location=device)
model = model.to(device)

def HomePage(request):
    return render (request,'home.html')

def SignupPage(request):
    if request.method=='POST':
        uname=request.POST.get('username')
        email=request.POST.get('email')
        pass1=request.POST.get('password1')
        pass2=request.POST.get('password2')

        if pass1!=pass2:
            return HttpResponse("Your password and confrom password are not Same!!")
        else:

            my_user=User.objects.create_user(uname,email,pass1)
            my_user.save()
            return redirect('login')
        



    return render (request,'signup.html')

def LoginPage(request):
    if request.method=='POST':
        username=request.POST.get('username')
        pass1=request.POST.get('pass')
        user=authenticate(request,username=username,password=pass1)
        if user is not None:
            login(request,user)
            return redirect('index')
        else:
            return HttpResponse ("Username or Password is incorrect!!!")

    return render (request,'login.html')

def LogoutPage(request):
    logout(request)
    return redirect('home')


@login_required(login_url='login')
def index(request):
    context={'a':1}
    return render(request, 'index.html')

def predictImage(request):
    if request.method == 'POST' and 'filePath' in request.FILES:
        # File handling
        fileObj = request.FILES['filePath']
        fs = FileSystemStorage()
        filePathName = fs.save(fileObj.name, fileObj)
        fileURL = fs.url(filePathName)
        filePath = os.path.join(settings.MEDIA_ROOT, filePathName)

        if not os.path.exists(filePath):
            return HttpResponse("File not found.")

        # Load and preprocess the image
        image = Image.open(filePath)

        # Convert to RGB if the image is not in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Predict the class
        image = image.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        class_names = {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented',3: 'Very_Mild_Demented'}
        # Dictionary of class names and their descriptions
        class_descriptions = {
            'MildDemented': 'This stage is characterized by mild cognitive decline. Individuals may experience slight memory lapses and changes in their thinking skills.usually still possible to function independently, but activities requiring concentration,organization may become noticeably more difficult.',
            'ModerateDemented': 'At this stage, individuals often have greater difficulty with memory and may start to need assistance with daily activities. There can be confusion, significant memory loss, and difficulty recognizing friends and family members. Problems with language, reasoning, sensory processing, and conscious thought can become evident.',
            'NonDemented': 'This label indicates no signs of dementia. The individual cognitive functions are within normal limits,their age there are no noticeable symptoms of memory loss , cognitive impairment.',
            'Very_Mild_Demented': 'This is often referred to as the pre-dementia or mild cognitive impairment stage. Symptoms are not severe enough to interfere with daily life and activities, but the individual may notice memory lapses like forgetting familiar words or the location of everyday objects.'
        }

        predicted_index = predicted.item()
        pred_label = class_names.get(predicted_index, "Unknown class")
        
        # Fetch the description for the predicted label
        pred_description = class_descriptions.get(pred_label, 'No description available.')

        # Prepare the context
        context = {
            'filePathName': fileURL, 
            'predictedLabel': pred_label, 
            'description': pred_description
        }
        
        # Render the response
        return render(request, 'index.html', context)

    # Handle case for non-POST requests
    return render(request, 'index.html')

#####################################
import torch
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import numpy as np
from .network import UNet
from django.core.files.storage import FileSystemStorage
from io import BytesIO

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = UNet().to(device)
unet.load_state_dict(torch.load('mod_unet_model.pth', map_location=device))
unet.eval()  # Set the model to evaluation mode

# Define the transform to be applied to the input image
transform = Compose([
    Resize((256, 256)),
    ToTensor(),
])

def preprocess_image(image):
    # Check if `image` is a PIL Image object
    if isinstance(image, Image.Image):
        image_pil = image
    else:
        # Convert `image` to string (assuming it's a path or other object) and open it with `Image.open()`
        image_pil = Image.open(str(image))
    
    # Convert the image to grayscale if necessary
    if image_pil.mode != 'L':
        image_pil = image_pil.convert('L')
    
    # Apply the defined transformations
    image_tensor = transform(image_pil).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(device)

def predict_segmentation(image_tensor):
    # Perform the segmentation
    with torch.no_grad():
        output = unet(image_tensor)
    
    # Convert the output to a numpy array
    output_array = output.squeeze().cpu().numpy()
    
    # Perform post-processing if necessary (e.g., thresholding)
    threshold = 0.1
    binary_output = np.where(output_array > threshold, 1, 0)
    
    return binary_output

def classify_segmentation(binary_mask):
    # Calculate the total number of pixels
    total_pixels = binary_mask.size
    
    # Calculate the number of pixels classified as affected
    affected_pixels = np.count_nonzero(binary_mask)
    
    # Calculate the percentage of affected area
    percentage_affected_area = (affected_pixels / total_pixels) * 100
    
    # Classify based on the percentage of affected area
    if percentage_affected_area < 20:
        classification = "Low"
    elif 20 <= percentage_affected_area < 50:
        classification = "Moderate"
    else:
        classification = "High"
    
    return classification, percentage_affected_area

def segmentImage(request):
    segmented_image_url = None
    classification_result = None
    
    if request.method == 'POST' and request.FILES.get('image'):
        # Retrieve the uploaded image from the request
        uploaded_image = request.FILES['image']
        
        # Load the uploaded image
        image = Image.open(uploaded_image)
        
        # Preprocess the image
        image_tensor = preprocess_image(image)
        
        # Predict segmentation
        segmented_mask = predict_segmentation(image_tensor)
        
        # Classify segmentation based on the percentage of affected area
        classification, percentage_affected_area = classify_segmentation(segmented_mask)
        classification_result = f"Classification: {classification}, Percentage Affected Area: {percentage_affected_area:.2f}%"
        
        # Convert binary mask to PIL image for visualization
        segmented_image = Image.fromarray((segmented_mask * 255).astype(np.uint8))
        
        # Save the segmented image to a BytesIO object
        with BytesIO() as segmented_image_io:
            segmented_image.save(segmented_image_io, format='PNG')
            segmented_image_io.seek(0)
            
            # Save the segmented image
            fs = FileSystemStorage()
            segmented_image_name = fs.save('segmented_image.png', segmented_image_io)
            segmented_image_url = fs.url(segmented_image_name)
    
    return render(request, 'segment_image.html', {'segmented_image_url': segmented_image_url, 'classification_result': classification_result})


#######################################
@login_required(login_url='login')
def index1(request):
    return render(request, 'index1.html')

def getPredictions(data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11):
    model = pickle.load(open('KNN.pkl', 'rb'))
    prediction = model.predict(np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11]]))
    return prediction[0]

def result(request):
    data1 = float(request.GET['Visit'])
    data2 = float(request.GET['MR Delay'])
    data3 = float(request.GET['M/F'])
    data4 = float(request.GET['Age'])
    data5 = float(request.GET[ 'EDUC'])
    data6 = float(request.GET['SES'])
    data7 = float(request.GET['MMSE'])
    data8 = float(request.GET[ 'CDR'])
    data9 = float(request.GET[ 'eTIV'])
    data10 = float(request.GET[ 'nWBV'])
    data11 = float(request.GET[ 'ASF'])

    result = getPredictions(data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11)
    return render(request, 'result.html', {'result': result})

