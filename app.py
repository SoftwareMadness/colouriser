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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

netG_A2B,netG_B2A,netD_A,netD_B = cyc.retrieve_nets()

netG_A2B = netG_A2B.to(device)
netG_B2A = netG_B2A.to(device)
netD_A = netD_A.to(device)
netD_B = netD_B.to(device)

pre_process = transforms.Compose([transforms.Resize(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                  ])

netG_B2A.eval()

img = input("Path to the image that you want to colourise:")

func = input("1.) Gray to color  2.) Color to gray, enter Number:")

net_f = netG_B2A

if func[0] == '2':
	net_f = netG_A2B

nimg = pre_process(Image.fromarray(cv2.imread(img))).unsqueeze(0)
calc = net_f(nimg.to(device)).to("cpu").detach()
vutils.save_image(calc, "result.png", normalize=True)
calc = cv2.imread("result.png")
cv2.imshow('frame', calc)
cv2.waitKey(0)
cv2.destroyAllWindows()
