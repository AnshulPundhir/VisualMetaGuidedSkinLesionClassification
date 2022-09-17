# Visual Meta-Guided Skin Lesion Classification

### This repo contains our work which has been accepted at ML-CDS 2022. Proceedings of our work shall be published in conjunction with the proceedings of MICCAI 2022.
### Co-authors: Anshul Pundhir, Ananya Agarwal, Saurabh Dadhich, Balasubramanian Raman

#### To run the code:

_folder_path:  Give path to working directory 

_model_name:   Name of the model and change import model accordingly while specifying the model 

Use main_pad  for PAD dataset and main_isic for ISIC dataset

Keep the flow of repository same to run the code

## Contents:
•	Overview of Project 
•	Dataset Description
•	Dependencies 
•	Problem-Solving Approach 
•	Results

## Overview of the Project: 
Detecting skin cancer in skin lesions is traditionally a costly and time-consuming task. Computer Aided Decision Support Systems can help in a timely cost-effective diagnosis. Traditionally such systems have relied on image-based data. We have leveraged multi-modal data comprising of images and meta-features of the patient. We have also created an attention mechanism which is using the metadata to guide visual attention for the images which helps focus on more relevant parts of the image.  

## Dataset Description:

We have tested our approach on 2 datasets PAD-UFES 20 and ISIC-19. These datasets containg images of different modalities (smartphone-captured and clinical images respectively) and also vary in the the number of metafeatures. Good performance on  both datasets proves our approach is effective. 

### Patients metadata is preprocessed same as MetaBlock (https://github.com/paaatcha/MetaBlock)    

#### PAD-UFES-20: https://data.mendeley.com/datasets/zr7vgbcyr2/1
It contains 2,298 clinical skin lesion images captured using smartphone devices. This dataset covers six lesion categories with 21 patient clinical details like gender, age, cancer history, etc. Skin lesion categories include Basal Cell Carcinoma, Melanoma, Squamous Cell Carcinoma, Actinic Keratosis, Nevus, and Seborrheic Keratosis. 

#### ISIC-19:     https://challenge.isic-archive.com/data/#2019
It is a large dataset containing 25,331 training and 8238 test dermoscopy images with 3 clinical features: Age, anatomical location, and gender. 
This dataset covers eight skin lesion categories: Melanocytic Nevus, Actinic Keratosis, Dermatofibroma, Squamous Cell Carcinoma, Melanoma, Basal Cell Carcinoma, Benign Keratosis, and Vascular Lesion.


## Dependencies 
These are mentioned in the Requirements file

## Problem-Solving Approach 

Our Approach can be summarized with the following architecture diagram-
 
![image](https://user-images.githubusercontent.com/79198655/190871162-e118a57b-b55f-4527-954e-29039675ec69.png)

More details can be found in our manuscript acctepted at ML-CDS 2022 (proceddings shall be published in conjucntion with MICCAI 2022, link shall be updated here after the proceedings are released).

## Results
We evaluated our approach on 5 CNN architectures- MobileNet, VGGNet, ResNet, EfficientNet, DenseNet and surpassed current SoTA reuslts on all architectures. 
More details can be found in our manuscript acctepted at ML-CDS 2022. 








