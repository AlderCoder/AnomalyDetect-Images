import torch
import torchvision
import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
from train_patch import *
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths
te_folder = "/  test/" # path to test scans
res_fake = "/results/fake/" # path to store reconstructed scans
res_real = "/results/real/" # path to store real scans
res_disp = "/results/disp/" # path to store disparity maps
labels = "/results/labels.csv" # path to the csv file with the labels of the scans

df = pd.read_csv(labels)

# Image parameters
act_size = 1120
p = 112

# Load model
model = Autoencoder(input_dim=(3, act_size, act_size))
model.load_state_dict(torch.load('model14.pth'))
model.to(device)
model.eval()

summary(model, input_size=(3, p, p))

def getPatches(folder, p):
    """
    This function divides all images in a folder into equal size patches of size pxp and return them as a numpy array.
    The pixel values are normalize by dividing through 255.
    """
    patches = []
    i_i = []
    i_j = []
    image_filenames = []
        
    doChunking = False

    index = 0
    i2 = 1
    for filename in os.listdir(folder):
        print(str(i2) + ", chunking test image '" + filename + "'")
        image_filenames.append(filename)
        
        i2 = i2 + 1
        image = Image.open(folder + filename)
        data = np.array(image)

        data = data.astype('float32') / 255.
        row, col,ch = data.shape
        
        for i in range(row):
            for j in range(col):
                if (i+1)*p <= row and (j+1)*p <= col:
                    patch = data[(i)*p:(i+1)*p,(j)*p:(j+1)*p,:]
                    patches.append(patch)
                    i_i.append(i)
                    i_j.append(j)
          
        if doChunking == True:
            if index >= 10:
                break
            else:
                index = index + 1
     
    patches = np.array(patches)
    
    return i_i, i_j, patches, image_filenames

  
i_i, i_j, x_test, image_filenames = getPatches(te_folder, p)

print(x_test.shape)
    
print("**********************Reconstructing Patches*******************")
decoded_imgs = []   

t, r, c, ch = x_test.shape

d_test = x_test.transpose((0, 3, 1, 2))

test_loader = DataLoader(d_test, batch_size=16, shuffle=False, num_workers=4)

for X in test_loader:
    X = X.to(device)
    x_pred = model(X)
    decoded_imgs.append(x_pred.detach().cpu().numpy())

decoded_imgs = np.concatenate(decoded_imgs, axis=0)
decoded_imgs = np.array(decoded_imgs)
decoded_imgs = decoded_imgs.transpose(0, 2, 3, 1)
print(decoded_imgs.shape)

d_imgs = []
d_test = []
heatmaps = []
i = 0
j = 0

img = np.zeros((act_size, act_size, 3), dtype='float32')
img2 = np.zeros((act_size, act_size, 3), dtype='float32')
img3 = np.zeros((act_size, act_size, 3), dtype='float32')

row, col,ch = img.shape
    
print("**********************Stitching Images*******************")
for k in range(len(i_i)):
    patch = decoded_imgs[k].reshape(p, p, 3);
    i = i_i[k]
    j = i_j[k]
    img[(i)*p:(i+1)*p,(j)*p:(j+1)*p,:] = patch
        
    img3[i*p:(i+1)*p,j*p:(j+1)*p,:] = x_test[k].reshape(p, p, 3)-patch 
        
    patch = x_test[k].reshape(p, p, 3);

    img2[i*p:(i+1)*p,j*p:(j+1)*p,:] = patch
        
    if i == 9 and j == 9:
        d_imgs.append(img)
        img = np.zeros((act_size, act_size, 3), dtype='float32') 
        d_test.append(img2)
        img2 = np.zeros((act_size, act_size, 3), dtype='float32')   
        heatmaps.append(img3)
        img3 = np.zeros((act_size, act_size, 3), dtype='float32') 

d_test = np.array(d_test)
d_imgs = np.array(d_imgs)
heatmaps = np.array(heatmaps)

print(d_test.shape)
print(d_imgs.shape)
print(heatmaps.shape)

t, r, c, ch = d_imgs.shape 
    
folder = res_fake
print("**********************Saving reconstructed images at " + folder + "*******************")
for i in range(t):
    A = (255 * d_imgs[i].reshape(act_size, act_size, 3)).astype(np.uint8)
    im = Image.fromarray(A)
    im.save(folder + image_filenames[i])

folder = res_real
print("**********************Saving real images at " + folder + "*******************")
for i in range(t):
    A = (255 * d_test[i].reshape(act_size, act_size, 3)).astype(np.uint8)
    im = Image.fromarray(A)
    im.save(folder + image_filenames[i])

folder = res_disp
print("**********************Saving disparity maps at " + folder + "*******************")
for i in range(t):
    A = (255 * heatmaps[i].reshape(act_size, act_size, 3)).astype(np.uint8)
    im = Image.fromarray(A)
    im.save(folder + image_filenames[i])
    sorted_heat = np.sort(heatmaps[i].flatten())[::-1][:500] # get the 500 largest pixel of the heatmap
    print("MSE: " + str(255*np.mean(heatmaps[i] * heatmaps[i])))
    df.loc[df['Filename'] == image_filenames[i], 'MSE'] = 255*np.mean(heatmaps[i] * heatmaps[i])
    print("Metric: " + str(255*np.mean(sorted_heat)))
    df.loc[df['Filename'] == image_filenames[i], 'Metric'] = 255*np.mean(sorted_heat)
df.to_csv("/results/labels.csv", index=False)
print("File is finished")
