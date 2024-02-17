from PIL import Image
from PIL import ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import ToTensor, Resize
from torch.utils.data.dataloader import default_collate
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm

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
            image = image.resize((480, 270)) # Gatekeeper
            #image = image.resiuze((500, 280)) # Town Hall Square

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


def make_autoencoder_timeline(test_loader, model, df, unterordner):
    """
    This function reconstructs images using an autoencoder model and calculates the corresponding reconstruction error for each day.
    The median of these errors is then saved in a CSV file with the rows labeled as “Day” and “MSE Conv”.
    """
    model.eval()
    criterion = nn.MSELoss()
    mse_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            if batch is None:
                continue
            img, filename = batch
            img = img.to(device)
            recon = model(img)
            mse = criterion(recon, img)
            mse_list.append(mse)


        mse_list_cpu = [mse.cpu().numpy() for mse in mse_list]
        avg_mse = 255*np.median(mse_list_cpu)
        print(f"Unterordner: {unterordner}, Durchschnittlicher Rekonstruktionsfehler: {avg_mse}")
        df.loc[df['Day'] == unterordner, 'MSE Conv'] = avg_mse
        df.to_csv("timeline.csv", index=False)

def make_pca_timeline(pca, test_folder, df, unterordner):
    """
    This function reconstructs images using PCA and calculates the corresponding reconstruction error for each day.
    The median of these errors is then saved in a CSV file with the rows labeled as “Day” and “MSE PCA”.
    """
    mse_list = []
    for i, filename in enumerate(os.listdir(test_folder)):
        try:
            image = Image.open(os.path.join(test_folder, filename))
            image = image.resize((480, 270))
            original = np.array(image).reshape(1, -1)
            transformed = pca.transform(original)
            reconstructed = pca.inverse_transform(transformed).astype(np.uint8)
            mse = mean_squared_error(original, reconstructed) / 10
            mse_list.append(mse)
        except UnidentifiedImageError:
            continue

    avg_mse = np.median(mse_list)
    print(np.std(mse_list))
    print(f"Unterordner: {unterordner}, Durchschnittlicher Rekonstruktionsfehler: {avg_mse}")
    df.loc[df['Day'] == unterordner, 'MSE PCA'] = avg_mse
    df.to_csv("timeline.csv", index=False)


if __name__ == '__main__':
    doAuto = True
    doPCA = True

    test_folder = "link_to_testimages" # with subfolders for each day 
    df = "timeline.csv"
    df = pd.read_csv(df)

    if doAuto:
        transform = transforms.ToTensor()

        model = Autoencoder_gate().to(device)
        model.load_state_dict(torch.load('gate.pth'))
        model.eval()

        for unterordner in sorted(os.listdir(test_folder)):
            unterordner_path = os.path.join(test_folder, unterordner)

            if os.path.isdir(unterordner_path):
                test = CustomDataset(folder_path=unterordner_path, transform=transform)
                test_loader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=my_collate_fn)

                make_autoencoder_timeline(test_loader, model, df, unterordner)

        print("File is finished!")

    if doPCA:
        pca = joblib.load('pca.pkl')

        for unterordner in sorted(os.listdir(test_folder)):
            unterordner_path = os.path.join(test_folder, unterordner)

            if os.path.isdir(unterordner_path):
                make_pca_timeline(pca, unterordner_path, df, unterordner)

        print("File is finished!")
