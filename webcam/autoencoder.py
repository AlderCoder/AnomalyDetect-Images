import torch
import torchvision
import os
import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        image = image.resize((480, 270)) # Gatekeeper
        image = image.resize((500, 280)) # Town Hall Square

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]

def save_image(output, save_path):
    output = (output.clamp(0, 1) * 255).to(torch.uint8).squeeze().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(output, mode='RGB')
    img.save(save_path)

# Model architecture for Gatekeeper
class Autoencoder_gate(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(240*135*4, 28),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(28, 240*135*4),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size = (4, 135, 240)),
            nn.ConvTranspose2d(4, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
 # Model architecture for Town Hall Square   
 class Autoencoder_town(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, stride=2, padding = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(125*70*4, 28),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(28, 125*70*4),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size = (4, 70, 125)),
            nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2, padding = 1, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding = 1, output_padding = 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_model(train_loader, val_loader, model):
    """
    This function trains an autoencoder model using the Mean Squared Error (MSE) loss function and evaluates it on a validation set.
    It returns the trained model.
    """
    criterion = nn.MSELoss()
    criterion1 = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters())
    num_epochs = 100
    
    # Defintion of the fixed backbone
    backbone = models.vgg16(pretrained=True, progress=True)
    backbone = backbone.features
    backbone.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for (img, _) in pbar:
                img = img.to(device)
                recon = model(img)
                loss = criterion(recon, img)
                # combined loss
                #back_img = backbone(img)
                #back_recon = backbone(recon)
                #loss1 = criterion1(back_recon, back_img)
                #loss2 = criterion(recon, img)
                #loss = 0.1*loss1 + 0.9*loss2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

        # Evaluate the model on the validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (img, _) in tqdm(valid_loader, desc=f'Validation'):
                img = img.to(device)
                recon = model(img)
                loss = criterion(recon, img)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(valid_loader)
            print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f} - Validation MSE: {255*avg_val_loss:.4f}")

    return model

def test_model(test_loader, model, df):
    """
    This function tests an autoencoder model. The original, reconstructed, and difference images are saved. 
    The reconstruction error is saved in a CSV file with the columns “Filename” and “MSE”.
    """
    model.eval()
    criterion = nn.MSELoss()

    with torch.no_grad():
        for img, filename in tqdm(test_loader, desc='Testing'):
            img = img.to(device)
            recon = model(img)
            mse = criterion(recon, img)
            disp = torch.abs(recon - img)

            df.loc[df['Filename'] == filename[0], 'MSE'] = 255*mse.item()
            print(f"MSE: {255*mse}")

            real_folder = "link_to_realfolder" #save the original image
            fake_folder = "link_to_fakefolder" #save the reconstructed image
            disp_folder = "link_to_dispfolder" #save the disparity map
            real_path = os.path.join(real_folder, filename[0])
            fake_path = os.path.join(fake_folder, filename[0])
            disp_path = os.path.join(disp_folder, filename[0])
            save_image(img, real_path)
            save_image(recon, fake_path)
            save_image(disp, disp_path)

        df.to_csv("results.csv", index=False)

if __name__ == '__main__':
    doTraining = True
    doTesting = False
    tr_folder = "link_to_trainfolder"
    valid_folder = "link_to_validationfolder"
    test_folder = "link_to_testfolder"
    results = "results.csv"
    results = pd.read_csv(results)

    transform = transforms.ToTensor()

    if doTraining == True:
        # Summary of the model
        auto = Autoencoder_gate().to(device)
        summary(auto, input_size=(3, 480, 270))
        #summary(auto, input_size=(3, 500, 280))

        # load the trainings and validation datasets
        train = datasets.ImageFolder(tr_folder, transform = transform)
        valid = datasets.ImageFolder(valid_folder, transform = transform)
        train_loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=8, pin_memory = True)
        valid_loader = DataLoader(valid, batch_size=16, shuffle=False, num_workers=8, pin_memory = True)

        # define a model and train it
        model = train_model(train_loader, valid_loader, auto)
        torch.save(model.state_dict(), 'gate.pth')
        print("Autoencoder-Modell wurde erfolgreich gespeichert!")

    if doTesting == True:
        # load the test dataset
        test = CustomDataset(folder_path=test_folder, transform=transform)
        test_loader = DataLoader(test, batch_size=1, shuffle=False)

        # load the trained model
        model = Autoencoder().to(device)
        model.load_state_dict(torch.load('gate.pth'))
        model.eval()

        test_model(test_loader, model, results)
        print("Testing is finished!")
