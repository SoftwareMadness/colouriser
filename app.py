import argparse
import itertools
import os
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils

from cyclegan_pytorch import DecayLR
from cyclegan_pytorch import Discriminator
from cyclegan_pytorch import Generator
from cyclegan_pytorch import ImageDataset
from cyclegan_pytorch import ReplayBuffer
from cyclegan_pytorch import weights_init

import cv2

import libcygtrn
from libcygtrn import CYCLES
from PIL import Image
import numpy as np

# define a video capture object

image_size = 256


cyc = CYCLES()

netG_A2B,netG_B2A,netD_A,netD_B = cyc.retrieve_nets()

pre_process = transforms.Compose([transforms.Resize(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                  ])

netG_B2A.eval()

img = input("Path to the image that you want to colourise:")

nimg = pre_process(Image.fromarray(cv2.imread(img))).unsqueeze(0)
calc = netG_B2A(nimg).detach()
vutils.save_image(calc, "result.png", normalize=True)
calc = cv2.imread("result.png")
cv2.imshow('frame', calc)
cv2.waitKey(0)
cv2.destroyAllWindows()
