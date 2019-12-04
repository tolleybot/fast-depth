#!/usr/bin/env python
# coding: utf-8

# In[38]:


import os
import time
import csv
import numpy as np

import torch
import torch.nn.parallel
import torch.optim
import models
import utils
from PIL import Image
import matplotlib.pyplot as plt


# In[2]:


checkpoint = torch.load('./mobilenet-nnconv5dw-skipadd-pruned.pth.tar',map_location=torch.device('cpu'))


# In[33]:


if type(checkpoint) is dict:
    start_epoch = checkpoint['epoch']
    best_result = checkpoint['best_result']
    model = checkpoint['model']
else:
    start_epoch = 0
    model = checkpoint


# In[41]:


def loadimg(filepath):
    img = Image.open(filepath).convert('RGB').resize((224,224),Image.NEAREST)
    img = np.asarray(img).astype('float')
    img /= 255.0
    img = np.expand_dims(img,axis=0)
    img = np.transpose(img, (0,3, 1, 2))
    return torch.from_numpy(img).float().to('cpu')


# In[72]:


img = loadimg('./examples/IMG_2148.png')


# In[73]:


with torch.no_grad():
    pred = model(img)


# In[74]:


pred[0][0]


# In[75]:


result = pred[0][0].numpy()


# In[76]:


from mpl_toolkits.mplot3d import Axes3D
# generate some sample data
import scipy.misc


# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:result.shape[0], 0:result.shape[1]]

# create the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, result ,rstride=1, cstride=1, cmap=plt.cm.gray,
        linewidth=0)

# show it
plt.show()


# In[ ]:




