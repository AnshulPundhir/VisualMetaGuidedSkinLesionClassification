import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torchvision import datasets, models, transforms
 
class resnet_isic(nn.Module): 
    def __init__(self, im_size, num_classes, attention = False):
        super(resnet_isic, self).__init__()
        self.im_size = im_size
        self.num_classes = num_classes
        self.attention = attention
        resnet_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
        self.model = models.resnet50(pretrained = True)

        for param in self.model.parameters(): 
            param.requires_grad = True

        self.initial = nn.Sequential(
                     self.model.conv1,
                     self.model.bn1,
                     self.model.relu,
                     self.model.maxpool,
                    )
        
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        self.avg_pool = self.model.avgpool 
     
        self.metadata_preprocessor  = nn.Sequential(
        nn.Linear(11, 81),
        nn.BatchNorm1d(81, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(True),
        nn.Dropout(0.2),
        nn.Linear(81, 512),
        nn.BatchNorm1d(512, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(True),
        nn.Dropout(0.2),
        nn.Linear(512, 2048),
        nn.BatchNorm1d(2048, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(True)) 

        # project to the query dimension
        self.projector1 = nn.Conv2d(256, 2048, kernel_size = 1, padding = 2, bias = False)
        self.projector2 = nn.Conv2d(512, 2048, kernel_size = 1, padding = 2, bias = False)
        self.projector3 = nn.Conv2d(1024, 2048, kernel_size = 1, padding = 2, bias = False)

        self.classifier_block = nn.Sequential(
                nn.Linear(2048*4, 2048, bias = False),
                nn.BatchNorm1d(2048, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(), 
                nn.Dropout(0.4),
                nn.Linear(2048, 512, bias = False),
                nn.BatchNorm1d(512, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, self.num_classes, bias = False)
            ) 


    def forward(self, input_x, metadata):   
        input_x =     self.initial(input_x)
        l1      =     self.layer1(input_x)            # 256
        l2      =     self.layer2(l1)                 # 512
        l3      =     self.layer3(l2)                 # 1024
        l4      =     self.layer4(l3)                 # 2048

        g_vector =    self.avg_pool(l4)  
  
        l1 = self.projector1(l1)
        l2 = self.projector2(l2)
        l3 = self.projector3(l3) 

        c1, g1 = self.linearAttentionBlock(l1, g_vector)
        c2, g2 = self.linearAttentionBlock(l2, g_vector)
        c3, g3 = self.linearAttentionBlock(l3, g_vector)

        g_meta = self.metadata_preprocessor(metadata) 
    
        g_vector = g_vector.reshape(g_vector.shape[0],g_vector.shape[1])
        g_vector = g_vector * g_meta

        x = torch.cat((g1, g2, g3, g_vector), dim = 1) 
        x = self.classifier_block(x)

        return [x, c1, c2, c3]


    def linearAttentionBlock(self, l, g,  normlize_method = "softmax"):
        N, C, W, H = l.size()
        #c = op(l + g) 
        c = (l * g).sum(dim=1)
        c = c.view(N,1,H,W)    #for dot product, op is 1x1 convolution.  
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