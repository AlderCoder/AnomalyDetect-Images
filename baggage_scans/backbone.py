import torch
import torchvision
import numpy as np
import torch.nn as nn
import cv2
import os
from torchvision import transforms
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
import lightly.data as data
from lightly.data import LightlyDataset
from lightly.utils.debug import std_of_l2_normalized
from torch.utils.data import TensorDataset, DataLoader

# The following transform function will return two augmented images per input image
# The transformation are describe on this site
# https://docs.lightly.ai/self-supervised-learning/lightly.transforms.html
transform = SimCLRTransform(input_size = 224, cj_prob = 0.0, random_gray_scale = 0.0, vf_prob = 0.5,
                            rr_prob = 0.5, rr_degrees = 90.0)

# Create a Lightly dataset from a foldeer with images
dataset = data.LightlyDataset(
    input_dir = "path_to_image_folder",
    transform=transform,
)


# Define dataloader
dataloader = DataLoader(
    dataset,               
    batch_size=256,
    shuffle=True,           # Shuffling is important!
    drop_last = True,
    num_workers=8,
)

# build a SimCLR model
class SimCLR(torch.nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, out_dim)

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

model = SimCLR(out_dim=128)

max_epochs = 50

# use a criterion (normalized temperature-scaled cross entropy loss) for self-supervised learning
criterion = NTXentLoss(temperature=0.5)

# define optimizer and scheduler to adjust learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.06, momentum = 0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    for epoch in range(max_epochs):
        for (x0, x1), _, _ in dataloader:

            x0 = x0.to(device)
            x1 = x1.to(device)
            model = model.to(device)

            z0 = model.forward(x0)
            z1 = model.forward(x1)

            loss = criterion(z0, z1)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            scheduler.step()

            representations = torch.cat([z0, z1], dim=0)
            representation_std = std_of_l2_normalized(representations)
            
            print(f"Epoch [{epoch}/{max_epochs}] Loss: {loss.item()}, representation std: {representation_std}")

    torch.save(model.state_dict(), 'backbone.pth')
    print("File is finished!")
