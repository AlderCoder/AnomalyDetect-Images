import torch
import torchvision
import os
import cv2
import csv
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
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data.dataloader import default_collate
from torchsummary import summary
import matplotlib.pyplot as plt

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
        try:
            image = Image.open(img_path).convert('RGB')
            if image.size != (480, 270):
                image = image.resize((480, 270)) # Gatekeeper
                #image = image.resize((500, 280)) # Town Hall Square

            if self.transform:
                image = self.transform(image)

            return image, self.image_files[idx]
        except UnidentifiedImageError:
            return None, img_path

def my_collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return None
    return default_collate(batch)

# Model architecture for Gatekeeper
class Autoencoder_gate(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=2, padding = 1),
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
            nn.ConvTranspose2d(4, 3, kernel_size=3, stride=2, padding = 1, output_padding = 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

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
        return decoded, encoded


def create_rep(data_loader, model):
    filenames = []
    reps = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Testing'):
            if batch is None:
                continue
            img, filename = batch
            img = img.to(device)
            recon, rep = model(img)
            reps.append(rep.detach().cpu().numpy().flatten())
            filenames.append(filename)
            
    pca = PCA(n_components=1)
    reps_pca = pca.fit_transform(reps)

    return reps_pca


if __name__ == '__main__':

    train_folder = 'folder_to_train_images'
    test_folder = "folder_to_test_images" #with the images of each day in a subfolder

    transform = transforms.ToTensor()
    train = CustomDataset(folder_path=train_folder, transform=transform)
    train_loader = DataLoader(train, batch_size=1, shuffle=False, collate_fn=my_collate_fn)
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load('/home/xadj/gate.pth'))
    model.eval()

    reps_pca = create_rep(train_loader, model)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(reps_pca)

    # Generate a range of values around your data for plotting
    X_plot = np.linspace(reps_pca.min(), reps_pca.max(), 1000)[:, np.newaxis]

    # Calculate the KDE for the plot values
    log_dens = kde.score_samples(X_plot)

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.fill(X_plot[:, 0], np.exp(log_dens))
    plt.title('Density Estimation of Training Bottleneck Representation')
    plt.xlabel('Bottleneck Representation')
    plt.ylabel('Density')

    # Save the plot as an image
    plt.savefig('train_density_gatekeeper.png')

    print("Trainings Dichte ist gespeichert!")
    plt.close()

    for unterordner in sorted(os.listdir(test_folder)):
            unterordner_path = os.path.join(test_folder, unterordner)
            print(unterordner_path)

            if os.path.isdir(unterordner_path):
                test = CustomDataset(folder_path=unterordner_path, transform=transform)
                test_loader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=my_collate_fn)

                reps = create_rep(test_loader, model)
                reps_pca = np.concatenate((reps_pca, reps))
                kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(reps_pca)

                X_plot = np.linspace(reps_pca.min(), reps_pca.max(), 1000)[:, np.newaxis]

                log_dens = kde.score_samples(X_plot)

                plt.figure(figsize=(10, 5))
                plt.fill(X_plot[:, 0], np.exp(log_dens))
                plt.title(f"Density Update as of {unterordner}")
                plt.xlabel('Bottleneck Representation')
                plt.ylabel('Density')

                plt.savefig(f'density_gatekeeper_{unterordner}.png')
                print(f"Dichte ist mit dem Tag {unterordner} aktualisiert!")
                plt.close()

    print("File is finished!")
