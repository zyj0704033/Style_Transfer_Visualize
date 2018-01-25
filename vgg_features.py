#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:30:03 2018

@author: sensetime
"""
import torch
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#from collections import namedtuple
import numpy as np

content_file = './0_ori.jpg'
style_file = './Patrick Earle.jpg'
im_size = 224
batch_size = 1

content_image = Image.open(content_file)
style_image = Image.open(style_file)
black_image = np.zeros((im_size,im_size,3))
black_image = Image.fromarray(black_image.astype('uint8'))
'''
#show the image
plt.figure()
plt.imshow(content_image)
plt.figure()
plt.imshow(style_image)
'''





class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        #vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        #out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]
'''    
vgg = Vgg16().cuda()
a,b,c =  content_image.size()
content_image = content_image.expand(1,a,b,c)
content_image = torch.autograd.Variable(content_image)
content_image = content_image.cuda()
content_feature_maps = vgg.forward(content_image)
'''
def content_loss_fromfeature(contfea1,contfea2):
    """
        :type contfea1,contfea2: ndarray(a,b,c,d) /feature map from vggnetwork
        :rtype: float /content loss
    """
    a,b,c,d = contfea1.shape
    print(a,b,c,d)
    feature1 = contfea1[0,:,:,:]
    feature2 = contfea2[0,:,:,:]
    return np.linalg.norm(feature1-feature2)/(b*c*d)

def content_loss_cal(image1,image2):
    """
        :type image1,image2:Image
        :rtype: float /content loss
    """
    transform = transforms.Compose([transforms.Scale(im_size),transforms.CenterCrop(im_size),transforms.ToTensor()])
    image1 = transform(image1)
    image2 = transform(image2)
    vgg = Vgg16().cuda()
    a,b,c = image1.size()
    image1 = torch.autograd.Variable(image1.expand(1,a,b,c))
    image2 = torch.autograd.Variable(image2.expand(1,a,b,c))
    image1 = image1.cuda()
    image2 = image2.cuda()
    image1_featuremap = vgg.forward(image1)
    image2_featuremap = vgg.forward(image2)
    for i in range(len(image1_featuremap)):
        print("layer%d:" %i)
        image1_feature = image1_featuremap[i].data.cpu().numpy()
        image2_feature = image2_featuremap[i].data.cpu().numpy()
        print(content_loss_fromfeature(image1_feature,image2_feature))
    

content_loss_cal(style_image,black_image)

    
''' visualize the feature map   
# h_relu1_2's feature map
feature_toshow = content_feature_maps[0].data.cpu().numpy()
plt.imshow(np.sum(feature_toshow[0,:,:,:],axis = 0).reshape(224,224))
'''