#Author:	Ananya Agarwal
#Email:		ananya_a@ch.iitr.ac.in
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torchvision import datasets, models, transforms

class vgg13_isic(nn.Module):
    def __init__(self, im_size, num_classes, attention = False):
        super(vgg13_isic, self).__init__()
        self.im_size = im_size
        self.num_classes = num_classes
        self.attention = attention

        self.model = models.vgg13_bn(pretrained=True)

        for param in self.model.parameters(): 
            param.requires_grad = True

        self.part1 = self.model.features[0:13]   #128
        self.part2 = self.model.features[13:20]   #256
        self.part3 = self.model.features[20:27]    #512
        self.part4 = self.model.features[27:]    #512
        self.adaptive_avg_pool_2d = nn.AdaptiveAvgPool2d((1,1))  

        # project to the query dimension
        self.projector1 = nn.Conv2d(128, 512, kernel_size = 1, padding = 2, bias = False)        
        self.projector2 = nn.Conv2d(256, 512, kernel_size = 1, padding = 2, bias = False)

        self.metadata_preprocessor  = nn.Sequential(
        nn.Linear(11, 81),
        nn.BatchNorm1d(81, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(True),
        nn.Dropout(0.2),
        nn.Linear(81, 256),
        nn.BatchNorm1d(256, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(True), 
        nn.Dropout(0.2),
        nn.Linear(256, 512),
        nn.BatchNorm1d(512, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(True)) 
        

        self.classifier_block = nn.Sequential(
        nn.Linear(512*4, 512),
        nn.ReLU(True),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes))


    def forward(self, input_x, metadata): 
        l1 = self.part1(input_x)
        l2 = self.part2(l1)
        l3 = self.part3(l2) 
        l4 = self.part4(l3)
        global_feat_vector= self.adaptive_avg_pool_2d(l4)

        metadata_pro = self.metadata_preprocessor(metadata)


        l1 = self.projector1(l1)
        l2 = self.projector2(l2)

        c1, g1 = self.linearAttentionBlock(l1, global_feat_vector)   #IMP- VGG HAD EARLIER OBTAINED HIGHER ACC THAN MOBILENET, MAYBE DUE TO SIGMOID INSTEAD OF SOFTMAX
        c2, g2 = self.linearAttentionBlock(l2, global_feat_vector)    #CAN ALSO CHECK WITH SIGOID THE NEXT TIME. 
        c3, g3 = self.linearAttentionBlock(l3, global_feat_vector)
        
        global_feat_vector = global_feat_vector.reshape(global_feat_vector.shape[0],global_feat_vector.shape[1])
        global_meta = global_feat_vector * metadata_pro   #this is giving us Gmeta
        
        g = torch.cat((g1, g2, g3, global_meta), dim = 1)

        x = self.classifier_block(g)

        return [x, c1, c2, c3]
    
    def linearAttentionBlock(self, l, g, normlize_method = "softmax"):
        N, C, H, W = l.size()
        c = (l * g).sum(dim=1).view(N,1,H,W)    #for dot product  
        if normlize_method == "softmax":
            a = F.softmax(c.view(N, 1, -1), dim =  2).view(N, 1, H, W)
        elif normlize_method == "sigmoid": 
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        
        if normlize_method == "softmax":
            g = g.view(N, C, -1).sum(dim = 2)
        elif normlize_method == "sigmoid":
            g = F.adaptive_avg_pool2d(g, (1, 1)).view(N, C)

        return c.view(N, 1, H, W), g