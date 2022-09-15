#Filename:	dataset.py


import numpy as np
import os
import torch
import pandas as pd
import torch.utils.data as Data
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

class CIFAR(Data.Dataset):

    def __init__(self, filename, labels,transform = None):
        self.transform = transform
        self.length = len(filename)
        self.images = filename
        self.labels = labels
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.25, iaa.Affine(scale={"x": (1.0, 2.0), "y": (1.0, 2.0)})),
            iaa.Scale((224,224)),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2),
            iaa.Sometimes(0.25, iaa.Affine(rotate=(-120, 120), mode='symmetric')),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),

            # noise
            iaa.Sometimes(0.1,
                          iaa.OneOf([
                              iaa.Dropout(p=(0, 0.05)),
                              iaa.CoarseDropout(0.02, size_percent=0.25)
                          ])),

            iaa.Sometimes(0.25,
                          iaa.OneOf([
                              iaa.Add((-15, 15), per_channel=0.5), # brightness
                              iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
                          ])),

        ])

        self.aug_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0),
            transforms.RandomHorizontalFlip(0.4),
            transforms.RandomRotation(50,expand=True),
            transforms.Resize((224,224)),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index): 
        image = Image.open(self.images[index]).convert("RGB")
        image = image.resize((224,224))
        image = self.aug.augment_image(np.array(image)).copy()
        label = self.labels[index] 

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class CIFAR_test(Data.Dataset):

    def __init__(self, filename, labels, transform = None):
        self.transform = transform 
        self.length = len(filename)
        self.images = filename
        self.labels = labels
        
    def __len__(self):
        return self.length

    def __getitem__(self, index): 
        image = Image.open(self.images[index]).convert("RGB")
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
        
if __name__ == "__main__":
    _base_path = "/home/anshul_p_cs/Desktop/Skin_lesion_project//PAD-UFES-20"    
    _imgs_folder_train = os.path.join(_base_path, "final_imgs_pad")
    _csv_path_train = os.path.join(_base_path, "pad-ufes-20_parsed_folders.csv")
    _csv_path_test = os.path.join(_base_path, "pad-ufes-20_parsed_test.csv")    
    train_csv_folder = pd.read_csv(_csv_path_train)
    
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])
    
    train_imgs_id = train_csv_folder['img_id'].values
    train_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id)[0:-3] + 'jpg' for img_id in train_imgs_id]
    train_labels = train_csv_folder['diagnostic_number'].values
    cifar100 = CIFAR(train_imgs_path, train_labels,train_transform) 
    
    train = Data.DataLoader(dataset = cifar100, batch_size = 10, shuffle = True, num_workers = 0)
    for epoch in range(1):
        for step, (batch_x, batch_y) in enumerate(train):
            print(step, batch_x.shape,'\n', batch_y)
            break

