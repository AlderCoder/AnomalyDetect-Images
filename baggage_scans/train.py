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
from backbone import SimCLR
from torchsummary import summary
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def add_gaussian_noise(image, mean=0, std=10):
    """
    Add gaussian noise to an image
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image should be a torch tensor.")

    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise

    return noisy_image

class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = sorted(os.listdir(folder_path))
        self.num_images = len(self.image_files)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

# Transformations for training and validation data
transform_train = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Lambda(lambda x: add_gaussian_noise(x, mean=0, std=(10**0.5)/255))
])

transform_valid = transforms.Compose([
    transforms.ToTensor(),
])

def save_losses_plot(train_losses, val_losses, save_path):
    """
    This function plot the evolution of trainings loss and validation loss
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(train_losses, label='Training Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    val_losses = torch.tensor(val_losses)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Validation MSE', color=color)  
    ax2.plot(val_losses.cpu().numpy(), label='Validation MSE', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.legend()
    plt.savefig(save_path)
    plt.close()

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
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
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


def train_model(train_loader, val_loader, patch_size):
    """
    Function for training the model
    """
    input_dim = (3, patch_size, patch_size)
    model = Autoencoder(input_dim)
    model.to(device)

    """
    #SimCLR backbone
    backbone = SimCLR(out_dim=128)
    backbone.load_state_dict(torch.load('/home/xadj/backbone_full1.pth'))
    backbone.to(device)
    backbone.eval()
    """

    """
    #Pre-trained VGG16 backbone
    backbone = models.vgg16(pretrained=True, progress=True)
    backbone = backbone.features
    backbone.to(device)
    backbone.eval()
    """

    n_epochs = 100
    train_losses = []
    val_losses = []

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
                """
                #Stylization Loss
                XX = backbone.backbone(X).flatten(start_dim=1)
                xp = backbone.backbone(x_pred).flatten(start_dim=1)
                loss1 = mae_loss(XX, xp)
                loss2 = mse_loss(X, x_pred)
                loss = 0.7*loss1 + 0.3*loss2
                """
                loss = mse_loss(X, x_pred)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
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
                loss = mse_loss(X_val, val_pred)
                val_loss += loss.item()
                val_mse += mse_loss(X_val, val_pred)
            avg_val_loss = val_loss / len(valid_loader)
            avg_mse_loss = val_mse / len(valid_loader)
            val_losses.append(255*avg_mse_loss)
            print(f"Epoch {epoch+1}/{n_epochs} - Validation Loss: {avg_val_loss:.4f} - Validation MSE: {255*avg_mse_loss:.4f}")

    save_losses_plot(train_losses, val_losses, 'loss.png')

    return model

def transfer_learning(train_loader, val_loader, model, patch_size):
    """
    function to use transfer learning
    """

    n_epochs = 50
    transfer_learning_losses = []

    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        model.train()  
        total_loss = 0.0
        with tqdm(train_loader, desc=f'Transfer Learning Epoch {epoch+1}/{n_epochs}') as pbar:
            for X in pbar:    
                X = X.to(device)
                x_pred = model.forward(X)
                loss = mse_loss(X, x_pred)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        transfer_learning_losses.append(avg_loss)
        print(f"Transfer Learning Epoch {epoch+1}/{n_epochs} - Average Loss: {avg_loss:.4f}")

        # Evaluate the model on the validation set
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Validation'):
                total_loss = 0.0
                X_val = batch.to(device)
                val_pred = model(X_val)
                loss = mse_loss(X_val, val_pred)
                val_loss += loss.item()
                val_mse += mse_loss(X_val, val_pred)
            avg_val_loss = val_loss / len(valid_loader)
            avg_mse_loss = val_mse / len(valid_loader)
            print(f"Transfer Learning Epoch {epoch+1}/{n_epochs} - Validation Loss: {avg_val_loss:.4f} - Validation MSE: {255*avg_mse_loss:.4f}")

    return model


if __name__ == '__main__':
    tr_folder = "link_to_train_folder"
    valid_folder = "link_to_valid_folder"
    act_size = 1120 #size of image

    # Create datasets and data loaders
    train_dataset = CustomDataset(folder_path=tr_folder, transform=transform_train)
    valid_dataset = CustomDataset(folder_path=valid_folder, transform=transform_valid)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory = True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory = True)

    # Output the architecture of the model
    auto = Autoencoder(input_dim=(3, act_size, act_size)).to(device)
    summary(auto, input_size=(3, act_size, act_size))

    # define a model and train it
    model = train_model(train_loader, valid_loader, act_size)
    torch.save(model.state_dict(), 'autoencoder.pth')
    print("Model is trained and saved!")

    """
    # Transfer Learning
    tr_folder_tl = "link_to_tl_train_folder"
    valid_folder_tl = "link_to_tl_valid_folder"
    model = Autoencoder(input_dim=(3, act_size, act_size)).to(device)
    model.load_state_dict(torch.load('autoencoder.pth'))

    train_dataset = CustomDataset(folder_path=tr_folder_tl, transform=transform_train)
    valid_dataset = CustomDataset(folder_path=valid_folder_tl, transform=transform_valid)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory = True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory = True)

    model = transfer_learning(train_loader, valid_loader, model, act_size)
    torch.save(model.state_dict(), 'autoencoder_tl.pth')
    print("Transfer Learning Model trained and saved!")
    """
