import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import joblib

def read_images(folder):
    """
    This function reads the images and store them in a numpy array.
    """
    images = []
    for filename in os.listdir(folder):
        image = Image.open(os.path.join(folder, filename))
        data = np.array(image)
        images.append(data.reshape(-1))
    return np.array(images)

def apply_pca(train_images, n_components):
    """
    This function applies the PCA.
    """
    pca = PCA(n_components=n_components)
    pca.fit(train_images)
    return pca

def apply_pca(pca, test_folder, csv_file):
    """
    This function applies the PCA to all images in the test folder. Additionaly the reconstructed images are stored and
    the corresponding reconstruction errors are saved in a csv file.
    """
    df = pd.read_csv(csv_file)
    for i, filename in enumerate(os.listdir(test_folder)):
        image = Image.open(os.path.join(test_folder, filename))
        original = np.array(image).reshape(1, -1)
        transformed = pca.transform(original)
        reconstructed = pca.inverse_transform(transformed)
        df.loc[df['Filename'] == filename, 'MSE'] = mean_squared_error(original, reconstructed)
        recon_image = Image.fromarray(reconstructed.reshape(image.size[1], image.size[0], 3).astype(np.uint8))
        recon_image.save(os.path.join('link_to_reconstructedfolder', filename))

    df.to_csv('path_to_csvfile', index=False)

    
if __name__ == '__main__':
    doTraining = True
    doTesting = True

    if doTraining:
        train_folder = 'link_to_trainingsfolder'
        
        train_images = read_images(train_folder)
        print('Training Images are loaded!')
        print(train_images.shape)
        pca = apply_pca(train_images, 28)
        print('fitting PCA is finished!')
        joblib.dump(pca, 'pca.pkl')
        print('PCA Model is saved and file is finished!')
        
    if doTesting:
        test_folder = 'link_to_testfolder'
        csv_file = 'link_to_csvfile' #the csv file to store the calculated MSE, rows: Filename, MSE
        
        pca = joblib.load('pca.pkl')
        apply_pca(pca, test_folder, csv_file)
        print("PCA is applied and file is finished!")
