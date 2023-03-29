import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image


dataroot = "./data/images/"
ngpu = 1
batch_size = 8
image_size = 64
seed_size = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 50000
lr_D = 0.00005
lr_G = 0.00005


transform1 = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 ])

transform2 = transforms.Compose([transforms.Resize(image_size),
                                 transforms.CenterCrop(image_size),
                                 transforms.RandomHorizontalFlip(p=1.0),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 ])

# transform3 = transforms.Compose([transforms.Resize(image_size),
#                                  transforms.CenterCrop(image_size),
#                                  transforms.ColorJitter(0.5, 0.5, 0.5),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                  ])

normal_dataset = dsets.ImageFolder(root=dataroot,
                                   transform=transform1,
                                   )

mirror_dataset = dsets.ImageFolder(root=dataroot,
                                   transform=transform2,
                                   )

color_jitter_dataset = dsets.ImageFolder(root=dataroot,
                                         transform=transform2,
                                         )

# dataset_list = [normal_dataset, mirror_dataset, color_jitter_dataset]
dataset_list = [normal_dataset, mirror_dataset]
dataset = torch.utils.data.dataset.ConcatDataset(dataset_list)
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True,
                                          )


police = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0, bias=False),
    nn.Flatten(),
    nn.Sigmoid(),
)


thief = nn.Sequential(
    nn.ConvTranspose2d(seed_size, 128, kernel_size=4, padding=0, stride=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),

    nn.ConvTranspose2d(128, 128, kernel_size=4, padding=1, stride=2, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),

    nn.ConvTranspose2d(128, 128, kernel_size=4, padding=1, stride=2, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),

    nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    nn.ConvTranspose2d(64, 3, kernel_size=4, padding=1, stride=2, bias=False),
    nn.Tanh()
)


# conv2d check
# netD = Police().to(device)
# sample, _ = normal_dataset[0]
# sample = sample.unsqueeze(0).to(device)
# print(sample.shape)
# print(netD(sample).shape)

# convTranspose2d check
# netG = Thief().to(device)
# noise = torch.randn(1, seed_size, 1, 1, device=device)
# print(noise.shape)
# print(netG(noise).shape)


police = police.to(device)
thief = thief.to(device)
criterion = nn.BCELoss()
optimizerD = optim.Adam(police.parameters(), lr=lr_D, betas=(0.9, 0.999))
optimizerG = optim.Adam(thief.parameters(), lr=lr_G, betas=(0.9, 0.999))

fixed_noise = torch.randn(batch_size, seed_size, 1, 1, device=device)
num_real_label = 1.
num_fake_label = 0.

epoch = 0
while True:
# for epoch in range(num_epochs):
    for i, (data, _) in enumerate(data_loader):

        police.zero_grad()

        optimizerD.zero_grad()
        optimizerG.zero_grad()

        img_real = data.to(device)
        # data[0] = img, data[1] = label(trash in this case)
        label = torch.full((batch_size,), num_real_label, dtype=torch.float, device=device)
        # 1.0 batch_size
        output = police(img_real).view(-1)
        # view(-1) : auto calculate dim, in this case: vector
        loss_real = criterion(output, label)
        loss_real.backward()

        noise = torch.randn(batch_size, seed_size, 1, 1, device=device)
        img_fake = thief(noise)
        output = police(img_fake).view(-1)
        label.fill_(num_fake_label)
        loss_fake = criterion(output, label)
        loss_fake.backward()


        thief.zero_grad()

        img_fake = thief(noise)
        output = police(img_fake).view(-1)
        label.fill_(num_real_label)
        loss_G = criterion(output, label)
        loss_G.backward()
        optimizerG.step()

        if loss_G < 2 * loss_real:
            optimizerD.step()

    if epoch % 50 == 0:
        print(f"{epoch}epoch done\nnetD:")
        with torch.no_grad():
            tmp_img = thief(fixed_noise)
            print(police(tmp_img))
        save_image(tmp_img, f"./fake/blockingPolice/img_fake{epoch}.jpg")
        torch.save(police.state_dict(), f"D{epoch}.pth")
        torch.save(thief.state_dict(), f"G{epoch}.pth")
    epoch += 1


# torch.save(netD.state_dict(), "D.pth")
# torch.save(netG.state_dict(), "G.pth")


# Article
#
# https://blog.jovian.com/pokegan-generating-fake-pokemon-with-a-generative-adversarial-network-f540db81548d
#
# https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types
#
# https://github.com/Zhenye-Na/pokemon-gan

