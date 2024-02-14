import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from torchsummary import summary
from train import *

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

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]

def test_model(model, test_loader, save_folder, df):
    """
    Function to test an autoencoder model
    MSE of original versus reconstructed image is calculated and store in a df
    Additional an other metric is claculated and store in a df
    """
    model.eval()
    mse_loss = nn.MSELoss()

    with torch.no_grad():
        for batch, filenames in tqdm(test_loader, desc='Testing'):
            X_test = batch.to(device)
            test_pred = model(X_test)

            for i in range(len(X_test)):
                original_image = X_test[i].cpu().numpy().transpose((1, 2, 0)) * 255
                reconstructed_image = test_pred[i].cpu().numpy().transpose((1, 2, 0)) * 255

                composite_image = np.concatenate([original_image, reconstructed_image], axis=1)
                composite_image = composite_image.astype(np.uint8)

                image_filename = os.path.join(save_folder, filenames[i])
                cv2.imwrite(image_filename, composite_image)

                # Calculate MSE
                mse = mse_loss(X_test[i], test_pred[i])
                print(f'MSE for image {i}: {mse.item()*255}')
                df.loc[df['Filename'] == filenames[i], 'MSE'] = mse.item()*255

                # Calculate average of the 500 largest deviating pixels
                sorted_diff = np.sort(np.abs(original_image - reconstructed_image).flatten())[::-1][:500]
                avg_deviation = np.mean(sorted_diff)
                print(f'Average deviation for image {i}: {avg_deviation}')
                df.loc[df['Filename'] == filenames[i], 'Metric'] = avg_deviation

        df.to_csv("results.csv", index=False)

if __name__ == '__main__':
    test_folder = "link_to_test_folder"
    save_folder = "link_to_folder_for_saving_reconstructed_images"
    labels = "results.csv"
    df = pd.read_csv(labels)

    act_size = 1120 # size of the image

    # Create datasets and data loaders
    transform = transforms.Compose([transforms.ToTensor(),])
    test_dataset = CustomDataset(folder_path=test_folder, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # Load a trained autoencoder model
    autoencoder = Autoencoder(input_dim=(3, act_size, act_size)).to(device)
    autoencoder.load_state_dict(torch.load('autoencoder.pth'))
    autoencoder.to(device)

    # Architecture of the autoencoder
    summary(autoencoder, input_size=(3, act_size, act_size))

    # Apply the test function
    test_model(autoencoder, test_loader, save_folder, df)
    print("File is finished!")
