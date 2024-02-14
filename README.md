# Master Thesis: Unsupervised Anomaly Detection in images

## Introduction
This repository contains the implementation of the Master Thesis: Unsupervised Anomaly Detection in images. In the baggage_scans folder are all code files for threat detection in X-ray scans. There are two files for training and testing with patches and a file for the training of a self-supervised SimCLR model which can serve as a backbone for the combined loss. In the webcam folder are all code file for the experiment on webcam images stored. The [autoencoder](https://github.zhaw.ch/xadj/MT-Anomaly/blob/main/webcam/autoencoder.py) and PCA file are for training the two reconstruction methods and the timeline and density files are for evaulating the two reconstrucion methods. 


## Configurations
Following libraries (version) and python 3.8.10 are used

1. torch 1.11.0
2. torchvision 0.12.0
3. numpy 1.21.0
4. tqdm 4.66.1
5. lightly 1.4.21

## Datasets
A confidental X-ray dataset is used but a public X-ray dataset can can be downloaded from the following URL:
[SIXray](https://github.com/MeioJane/SIXray) 

Two webcam sources are used for the thesis. It is possible to use different webcam data as long as they have a temporal structure.

## Steps for baggage scans

1. Load the desired dataset (e. g. SIXray) into the dataset folder. The dataset hierarchy is proposed as follows:

```
├── dataset
│   ├── dataset name (e.g. sixray)
│   │   └── train
│   │   └── valid
│   │   └── test
│   │   └── results
│   │   │   └── disp
│   │   │   └── fake
│   │   │   └── real
│   │   │   └── results
│   │   │   └── labels.csv
```

2) For the training step, please provide the normal samples of the dataset and split them into ‘train’ and ‘valid’ folder.

3) For the test step, please provide test (abnormal and normal) samples in ‘test’ folder and create a csv file with the labels.

4) Please run either the ‘train.py’ or ‘train_patch.py’ to train a model.

5) Afterward, please run the ‘test.py’ or ‘test_patch.py’ to produce reconstructions, disparity maps and adding the MSE and a classification metric to the labels.csv 

6) Now it is possible to classify the X-ray scans based on either MSE or on the recommended alternative metric.

## Steps for webcam images

1. Load the desired dataset with webcam images into the dataset folder. The dataset hierarchy is proposed as follows:

```
├── dataset
│   ├── dataset name (e. g. Gate)
│   │   └── train
│   │   └── valid
│   │   └── images
│   │   │   └── 2022-07-28
│   │   │   └── 2022-07-29
│   │   │   └── 2022-07-30
│   │   │   └── ...
│   │   │   └── timeline.csv
│   │   └── density
```

2) For the training step, please provide some webcam images of the dataset and split them into ‘train’ and ‘valid’ folder.

3) For the evaluation step, please provide the webcam images in chronological order (in this case days) in the ‘images’ folder.

4) Please run ‘pca.py’ and ‘autoencoder.py’ two train the two reconstructed methods.

5) For the temporal evaluation of the reconstruction error, run ‘timeline.py’ to fill out the timeline.csv with values.

6) For the investigation of the bottneck represnetation density, run ‘density.py’ to save density images in the ‘density’ folder.

## Results
The results of the experiment with the webcam images are presented in the '…/webcam/csv/' folder. 

## Citation
If you use the proposed framework (or any part of this code in your research), please cite the following:

```
@mastersthesis{alder2024anomaly,
  author = {Alder, Joel},
  title = {Unsupervised Anomaly Detection in images},
  school = {Swiss Federal Institute of Technology Zurich (ETH)},
  year = {2024},
  type = {Masterthesis}
}
```

## Contact
If you have any query, please feel free to contact: xadj@zhaw.ch
