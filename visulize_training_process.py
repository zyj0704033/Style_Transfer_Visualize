#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:35:56 2018

@author: sensetime
"""
import numpy as np
import matplotlib.pyplot as plt

content_loss =  np.loadtxt('./content_loss.txt')
style_loss = np.loadtxt('./style_loss.txt')
total_loss = np.loadtxt('./total_loss.txt')

plt.figure('content loss')
plt.title('content loss')
plt.plot(range(len(content_loss)),content_loss)
plt.show()

plt.figure('style loss')
plt.title('style loss')
plt.plot(range(len(style_loss)),style_loss)
plt.show()

plt.figure('total loss')
plt.title('total loss')
plt.plot(range(len(total_loss)),total_loss)
plt.show()