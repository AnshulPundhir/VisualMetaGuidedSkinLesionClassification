def linearAttentionBlock_metadata(self, l, g, metadata,metadata_preprocessor, normlize_method = "softmax"):
    N, C, H, W = l.size()
    c = (l * g).sum(dim=1).view(N,1,H,W)    #for dot product  
    meta_pro = metadata_preprocessor(metadata)

    if normlize_method == "softmax":
        a = F.softmax(c.view(N, 1, -1), dim =  2).view(N, 1, H, W)
    elif normlize_method == "sigmoid": 
        a = torch.sigmoid(c)

    meta_pro = meta_pro.reshape((meta_pro.shape[0],meta_pro.shape[1],1,1))
    tan_fn = nn.Tanh() 
    a_dash = tan_fn(a.expand_as(l) * meta_pro)

    g = torch.mul(a_dash, l)
    
    if normlize_method == "softmax":
        g = g.view(N, C, -1).sum(dim = 2)
    elif normlize_method == "sigmoid":
        g = F.adaptive_avg_pool2d(g, (1, 1)).view(N, C)

    return c.view(N, 1, H, W), g