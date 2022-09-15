#Filename:	custom_dataset.py
#Institute: IIT Roorkee

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from imgaug import augmenters as iaa
 
class meta_img_dataset(Dataset):

    def __init__(self, filename, metadata, labels, transform = None):
        self.transform = transform 
        self.length = len(filename)
        self.images = filename
        self.labels = labels
        self.metadata=metadata
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

    def __len__(self):
        return self.length

    def __getitem__(self, index): 
        image = Image.open(self.images[index]).convert("RGB")
        image = self.aug.augment_image(np.array(image)).copy()
        label = self.labels[index]
        metadata = self.metadata[index] 

        if self.transform is not None:
            image = self.transform(image)

        return image, metadata, label

class meta_img_dataset_test(Dataset):

    def __init__(self, filename, metadata, labels, transform = None):
        self.transform = transform 
        self.length =    len(filename)
        self.images =    filename
        self.labels =    labels
        self.metadata =  metadata
        
    def __len__(self):
        return self.length

    def __getitem__(self, index): 
        image = Image.open(self.images[index]).convert("RGB")
        label = self.labels[index]
        metadata = self.metadata[index] 

        if self.transform is not None:
            image = self.transform(image)

        return image, metadata, label

