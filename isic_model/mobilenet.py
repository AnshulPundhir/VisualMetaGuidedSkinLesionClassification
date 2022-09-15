import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torchvision import datasets, models, transforms

class mobile_isic(nn.Module):
    def __init__(self, im_size, num_classes,attention = False):
        super(mobile_isic, self).__init__()
        self.im_size = im_size
        self.num_classes = num_classes
        self.attention = attention

        self.model = models.mobilenet_v2(pretrained=True)

        for param in self.model.parameters(): 
            param.requires_grad = True

        self.part1 = self.model.features[0:10]        # 64,14,14
        self.part2 = self.model.features[10:13]       # 96, 14, 14
        self.part3 = self.model.features[13:16]       # 160, 7, 7 
        self.part4 = self.model.features[16:]         # 1280, 7, 7

        self.adaptive_avg_pool_2d = nn.AdaptiveAvgPool2d((1,1))

        self.projector1 = nn.Conv2d(64, 1280, kernel_size = 1, padding = 0)
        self.projector2 = nn.Conv2d(96, 1280, kernel_size = 1, padding = 0)
        self.projector3 = nn.Conv2d(160, 1280, kernel_size = 1, padding = 0)
 
        self.metadata_preprocessor  = nn.Sequential(
        nn.Linear(11, 81),
        nn.BatchNorm1d(81, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(True),
        nn.Dropout(0.2),
        nn.Linear(81, 512),
        nn.BatchNorm1d(512, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(True),
        nn.Dropout(0.2),
        nn.Linear(512, 1280),
        nn.BatchNorm1d(1280, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(True)) 
 
        self.classifier_block = nn.Sequential(
        nn.Linear(1280*4, 1280),
        nn.ReLU(True),
        nn.Dropout(0.3),
        nn.Linear(1280, 512),
        nn.ReLU(True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
        )

    def forward(self, input_x, metadata): 
        l1 = self.part1(input_x)
        l2 = self.part2(l1)
        l3 = self.part3(l2)
        l4 = self.part4(l3)
        global_feat_vector = self.adaptive_avg_pool_2d(l4)
        
        l1 = self.projector1(l1)
        l2 = self.projector2(l2)
        l3 = self.projector3(l3)

        c1, g1 = self.linearAttentionBlock(l1, global_feat_vector)
        c2, g2 = self.linearAttentionBlock(l2, global_feat_vector)
        c3, g3 = self.linearAttentionBlock(l3, global_feat_vector)

        g_meta = self.metadata_preprocessor(metadata) 

        global_feat_vector = global_feat_vector.reshape(global_feat_vector.shape[0],global_feat_vector.shape[1])
        global_feat_vector = global_feat_vector * g_meta

        g = torch.cat((g1, g2, g3, global_feat_vector), dim = 1)
        x = self.classifier_block(g)

        return [x, c1, c2, c3]

    def linearAttentionBlock(self, l, g, normlize_method = "softmax"):      
        N, C, H, W = l.size() 
        c = (l * g).sum(dim=1)
        c = c.view(N,1,H,W)    #for dot product, op is 1x1 convolution.  
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