import torch
import torchvision
import os
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
#from backbone import SimCLR
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getPatches(folder, isTraining, p):
    """
    This function divides all images in a folder into equal size patches of size pxp and return them as a numpy array.
    Optional if isTraining=True then Gaussian noise is added to the patches.
    The pixel values are normalize by dividing through 255.
    """
    mean = 0
    var = 10
    sigma = var ** 0.5
    act_size = 2240
    gaussian = np.random.normal(mean, sigma, (act_size, act_size))
    patches = []
        
    doChunking = False

    index = 0
    i2 = 1
    for filename in os.listdir(folder):
        if isTraining == True:
            print(str(i2) + ", chunking training image '" + filename + "'")
        else:
            print(str(i2) + ", chunking validation image '" + filename + "'")
        
        i2 = i2 + 1
        image = Image.open(folder + filename)
        data = np.array(image)

        if isTraining == True:
                # adding Gaussian noise            
                if len(data.shape) == 2:
                    data = data + gaussian
                else:
                    data[:, :, 0] = data[:, :, 0] + gaussian
                    data[:, :, 1] = data[:, :, 1] + gaussian
                    data[:, :, 2] = data[:, :, 2] + gaussian

        data = data.astype('float32') / 255.
        row, col,ch = data.shape
        
        for i in range(row):
            for j in range(col):
                if (i+1)*p <= row and (j+1)*p <= col:
                    patch = data[(i)*p:(i+1)*p,(j)*p:(j+1)*p,:]
                    patches.append(patch)
         
        if doChunking == True:
            if index >= 10:
                break
            else:
                index = index + 1
     
    patches = np.array(patches)
    
    return patches

class Autoencoder(nn.Module):
    """
    The Autoencoder model.
    """
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=7)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=7, mode='bilinear', align_corners=False),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_model(train_loader, val_loader, p):
    """
    This function trains the autoencdoer model.
    Optionally a combined loss with either self-supervised SimCLR or pre-trained VGG16 backbone can be used.
    """

    model = Autoencoder(input_dim=(3, p, p))
    model.to(device)

    #self-trained SimCLR backbone
    #backbone = SimCLR(out_dim=128)
    #backbone.load_state_dict(torch.load('backbone.pth'))
    #backbone.to(device)
    #backbone.eval()

    # pre-trained VGG16 backbone
    #backbone = models.vgg16(pretrained=True, progress=True)
    #backbone = backbone.features
    #backbone.to(device)

    n_epochs = 100

    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(n_epochs): 
        model.train()   
        total_loss = 0.0
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}') as pbar:
            for X in pbar:    
                X = X.to(device)
                x_pred = model.forward(X)
                #XX = backbone.backbone(X).flatten(start_dim=1)
                #xp = backbone.backbone(x_pred).flatten(start_dim=1)
                #loss1 = mae_loss(XX, xp)
                #loss2 = mse_loss(X, x_pred)
                #loss = 0.7*loss1 + 0.3*loss2
                loss = mse_loss(X, x_pred)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs} - Average Loss: {avg_loss:.4f}")

        # Evaluate the model on the validation set
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Validation'):
                total_loss = 0.0
                X_val = batch.to(device)
                val_pred = model(X_val)
                #XX_val = backbone(X_val)
                #xp_val = backbone(val_pred)
                #loss1 = mae_loss(XX_val, xp_val)
                #loss2 = mse_loss(X_val, val_pred)
                #loss = 0.7*loss1 + 0.3*loss2
                loss = mse_loss(X_val, val_pred)
                val_loss += loss.item()
                val_mse += mse_loss(X_val, val_pred)
            avg_val_loss = val_loss / len(valid_loader)
            avg_mse_loss = val_mse / len(valid_loader)
            print(f"Epoch {epoch+1}/{n_epochs} - Validation Loss: {avg_val_loss:.4f} - Validation MSE: {255*avg_mse_loss:.4f}")

    return model


if __name__ == '__main__':
    doPreProcess = False
    tr_folder = "/home/xadj/models/datasets/casra/train/"
    valid_folder = "/home/xadj/models/datasets/casra/valid/"
    
    # image parameters
    act_size = 1120
    p = 112

    # load the training and testing data
    if doPreProcess:
        x_train = getPatches(tr_folder, False, p)
        x_valid = getPatches(valid_folder, False, p)
        np.save("train_data.npy", x_train)
        print("train data is preprocessed and saved!")
        np.save("valid_data.npy", x_valid)
        print("validation data is preprocessed and saved!")
    else:
        x_train = np.load("train_data.npy")
        print("train data is loaded!")
        x_valid = np.load("valid_data.npy")
        print("validation data is loaded!")

    print(x_train.shape)
    print(x_valid.shape)
    x_train = x_train.transpose((0, 3, 1, 2))  # Transpose to (16, 3, 120, 120)
    x_valid = x_valid.transpose((0, 3, 1, 2))  # Transpose to (16, 3, 120, 120)

    # Create data loaders for the training and valid data
    train_loader = DataLoader(x_train, batch_size=16, shuffle = True, num_workers = 8, pin_memory = True)
    valid_loader = DataLoader(x_valid, batch_size=16, shuffle = False, num_workers = 8, pin_memory = True)

    # Summary of the model
    auto = Autoencoder(input_dim=(3, act_size, act_size)).to(device)
    summary(auto, input_size=(3, p, p))

    # define a model and train it
    model = train_model(train_loader, valid_loader, p)
    torch.save(model.state_dict(), 'autoencoder.pth')
    print("autoencoder model is saved!")
