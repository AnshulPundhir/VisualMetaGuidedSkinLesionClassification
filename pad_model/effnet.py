import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torchvision import datasets, models, transforms

class effnet_pad(nn.Module):
    def __init__(self, im_size, num_classes, attention = False):
        super(effnet_pad, self).__init__()
        self.im_size = im_size
        self.num_classes = num_classes
        self.attention = attention

        self.model = models.efficientnet_b4(pretrained = True)  
        for param in self.model.parameters(): 
            param.requires_grad = True        
 
        self.part1    =  self.model.features[0:7] 
        self.part2    = list(self.model.features[7][0].children())[0][0:3]    # 1632
        self.part3    = list(self.model.features[7][0].children())[0][3:]  
        self.part4    = list(self.model.features[7][0].children())[1] 
        self.part5    = list(self.model.features[7][1].children())[0][0:3]    # 2688
        self.part6    = nn.Sequential(
                            list(self.model.features[7][1].children())[0][3:],
                            list(self.model.features[7][1].children())[1],
                            self.model.features[8:]
        )

        self.adaptive_avg_pool_2d = self.model.avgpool

        # project to the query dimension
        self.projector1 = nn.Conv2d(1632, 1792, kernel_size = 1, padding = 2, bias = False)
        self.projector2 = nn.Conv2d(2688, 1792, kernel_size = 1, padding = 2, bias = False)

        self.classifier_block = nn.Sequential(
        nn.Linear(1792*3, 1792, bias = False),
        nn.BatchNorm1d(1792, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(), 
        nn.Dropout(0.4),      
        nn.Linear(1792, 512, bias = False),
        nn.BatchNorm1d(512, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, self.num_classes, bias = False)
        )
  
        self.metadata_preprocessor  = nn.Sequential(
        nn.Linear(81, 512),
        nn.BatchNorm1d(512, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(True),
        nn.Dropout(0.2),
        nn.Linear(512, 1792),
        nn.BatchNorm1d(1792, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(True)) 

    def forward(self, input_x, metadata):
        input_x      = self.part1(input_x)
        l1           = self.part2(input_x)
        input_x      = self.part3(l1)            
        input_x      = self.part4(input_x)
        l2           = self.part5(input_x)
        input_x      = self.part6(l2)
         
        g_vector = self.adaptive_avg_pool_2d(input_x)

        l1 = self.projector1(l1)
        l2 = self.projector2(l2)

        c1, g1 = self.linearAttentionBlock(l1, g_vector)
        c2, g2 = self.linearAttentionBlock(l2, g_vector)

        g_meta = self.metadata_preprocessor(metadata) 

        g_vector = g_vector.reshape(g_vector.shape[0],-1)
        g_vector = g_vector * g_meta 

        g = torch.cat((g1, g2, g_vector), dim = 1)
        x = self.classifier_block(g)

        return [x, c1, c2]

    def linearAttentionBlock(self, l, g, normlize_method = "softmax"):
        N, C, W, H = l.size()
        c = (l * g).sum(dim=1).view(N,1,H,W)    #for dot product  
        if normlize_method == "softmax":
            a = F.softmax(c.view(N, 1, -1), dim =  2).view(N, 1, W, H)
        elif normlize_method == "sigmoid":
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        
        if normlize_method == "softmax":
            g = g.view(N, C, -1).sum(dim = 2)
        elif normlize_method == "sigmoid":
            g = F.adaptive_avg_pool2d(g, (1, 1)).view(N, C)

        return c.view(N, 1, W, H), g