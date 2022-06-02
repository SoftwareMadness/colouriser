# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import itertools
import os
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from cyclegan_pytorch import DecayLR
from cyclegan_pytorch import Discriminator
from cyclegan_pytorch import Generator
from cyclegan_pytorch import ImageDataset
from cyclegan_pytorch import ReplayBuffer
from cyclegan_pytorch import weights_init

class CYCLES:

    def train(self,manualSeed = None,cuda = True,dataroot="data",dataset_p="dataset",image_size=256,batch_size=1,netG_A2B_f="",netG_B2A_f="",netD_A_f="",netD_B_f="",epochs = 100,decay_epochs = 50,outf="./outputs",print_freq=10,learning_rate=0.0002):

        try:
            os.makedirs(outf)
        except OSError:
            pass

        try:
            os.makedirs("weights")
        except OSError:
            pass

        if manualSeed is None:
            manualSeed = random.randint(1, 10000)
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        cudnn.benchmark = True

        if torch.cuda.is_available() and not cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        # Dataset
        dataset = ImageDataset(root=os.path.join(dataroot, dataset_p),
                               transform=transforms.Compose([
                                   transforms.Resize(int(image_size * 1.12), Image.BICUBIC),
                                   transforms.RandomCrop(image_size),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                               unaligned=True)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        try:
            os.makedirs(os.path.join(outf, dataset_p, "A"))
            os.makedirs(os.path.join(outf, dataset_p, "B"))
        except OSError:
            pass

        try:
            os.makedirs(os.path.join("weights", dataset_p))
        except OSError:
            pass

        device = torch.device("cuda:0" if cuda else "cpu")

        # create model
        netG_A2B = Generator().to(device)
        netG_B2A = Generator().to(device)
        netD_A = Discriminator().to(device)
        netD_B = Discriminator().to(device)

        netG_A2B.apply(weights_init)
        netG_B2A.apply(weights_init)
        netD_A.apply(weights_init)
        netD_B.apply(weights_init)

        if netG_A2B_f != "":
            netG_A2B.load_state_dict(torch.load(netG_A2B))
        if netG_B2A_f != "":
            netG_B2A.load_state_dict(torch.load(netG_B2A))
        if netD_A_f != "":
            netD_A.load_state_dict(torch.load(netD_A))
        if netD_B_f != "":
            netD_B.load_state_dict(torch.load(netD_B))

        # define loss function (adversarial_loss) and optimizer
        cycle_loss = torch.nn.L1Loss().to(device)
        identity_loss = torch.nn.L1Loss().to(device)
        adversarial_loss = torch.nn.MSELoss().to(device)

        # Optimizers
        optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                       lr=learning_rate, betas=(0.5, 0.999))
        optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        lr_lambda = DecayLR(epochs, 0, decay_epochs).step
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lr_lambda)
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lr_lambda)

        g_losses = []
        d_losses = []

        identity_losses = []
        gan_losses = []
        cycle_losses = []

        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()

        for epoch in range(0, epochs):
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
            for i, data in progress_bar:
                # get batch size data
                real_image_A = data["A"].to(device)
                real_image_B = data["B"].to(device)
                batch_size = real_image_A.size(0)

                # real data label is 1, fake data label is 0.
                real_label = torch.full((batch_size, 1), 1, device=device, dtype=torch.float32)
                fake_label = torch.full((batch_size, 1), 0, device=device, dtype=torch.float32)

                ##############################################
                # (1) Update G network: Generators A2B and B2A
                ##############################################

                # Set G_A and G_B's gradients to zero
                optimizer_G.zero_grad()

                # Identity loss
                # G_B2A(A) should equal A if real A is fed
                identity_image_A = netG_B2A(real_image_A)
                loss_identity_A = identity_loss(identity_image_A, real_image_A) * 5.0
                # G_A2B(B) should equal B if real B is fed
                identity_image_B = netG_A2B(real_image_B)
                loss_identity_B = identity_loss(identity_image_B, real_image_B) * 5.0

                # GAN loss
                # GAN loss D_A(G_A(A))
                fake_image_A = netG_B2A(real_image_B)
                fake_output_A = netD_A(fake_image_A)
                loss_GAN_B2A = adversarial_loss(fake_output_A, real_label)
                # GAN loss D_B(G_B(B))
                fake_image_B = netG_A2B(real_image_A)
                fake_output_B = netD_B(fake_image_B)
                loss_GAN_A2B = adversarial_loss(fake_output_B, real_label)

                # Cycle loss
                recovered_image_A = netG_B2A(fake_image_B)
                loss_cycle_ABA = cycle_loss(recovered_image_A, real_image_A) * 10.0

                recovered_image_B = netG_A2B(fake_image_A)
                loss_cycle_BAB = cycle_loss(recovered_image_B, real_image_B) * 10.0

                # Combined loss and calculate gradients
                errG = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

                # Calculate gradients for G_A and G_B
                errG.backward()
                # Update G_A and G_B's weights
                optimizer_G.step()

                ##############################################
                # (2) Update D network: Discriminator A
                ##############################################

                # Set D_A gradients to zero
                optimizer_D_A.zero_grad()

                # Real A image loss
                real_output_A = netD_A(real_image_A)
                errD_real_A = adversarial_loss(real_output_A, real_label)

                # Fake A image loss
                fake_image_A = fake_A_buffer.push_and_pop(fake_image_A)
                fake_output_A = netD_A(fake_image_A.detach())
                errD_fake_A = adversarial_loss(fake_output_A, fake_label)

                # Combined loss and calculate gradients
                errD_A = (errD_real_A + errD_fake_A) / 2

                # Calculate gradients for D_A
                errD_A.backward()
                # Update D_A weights
                optimizer_D_A.step()

                ##############################################
                # (3) Update D network: Discriminator B
                ##############################################

                # Set D_B gradients to zero
                optimizer_D_B.zero_grad()

                # Real B image loss
                real_output_B = netD_B(real_image_B)
                errD_real_B = adversarial_loss(real_output_B, real_label)

                # Fake B image loss
                fake_image_B = fake_B_buffer.push_and_pop(fake_image_B)
                fake_output_B = netD_B(fake_image_B.detach())
                errD_fake_B = adversarial_loss(fake_output_B, fake_label)

                # Combined loss and calculate gradients
                errD_B = (errD_real_B + errD_fake_B) / 2

                # Calculate gradients for D_B
                errD_B.backward()
                # Update D_B weights
                optimizer_D_B.step()

                progress_bar.set_description(
                    f"[{epoch}/{epochs - 1}][{i}/{len(dataloader) - 1}] "
                    f"Loss_D: {(errD_A + errD_B).item():.4f} "
                    f"Loss_G: {errG.item():.4f} "
                    f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.4f} "
                    f"loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.4f} "
                    f"loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB).item():.4f}")

                if i % print_freq == 0:
                    vutils.save_image(real_image_A,
                                      f"{outf}/{dataset_p}/A/real_samples_epoch_{epoch}_{i}.png",
                                      normalize=True)
                    vutils.save_image(real_image_B,
                                      f"{outf}/{dataset_p}/B/real_samples_epoch_{epoch}_{i}.png",
                                      normalize=True)

                    fake_image_A = 0.5 * (netG_B2A(real_image_B).data + 1.0)
                    fake_image_B = 0.5 * (netG_A2B(real_image_A).data + 1.0)

                    vutils.save_image(fake_image_A.detach(),
                                      f"{outf}/{dataset_p}/A/fake_samples_epoch_{epoch}_{i}.png",
                                      normalize=True)
                    vutils.save_image(fake_image_B.detach(),
                                      f"{outf}/{dataset_p}/B/fake_samples_epoch_{epoch}_{i}.png",
                                      normalize=True)

            # do check pointing
            torch.save(netG_A2B.state_dict(), f"weights/{dataset_p}/netG_A2B_epoch_{epoch}.pth")
            torch.save(netG_B2A.state_dict(), f"weights/{dataset_p}/netG_B2A_epoch_{epoch}.pth")
            torch.save(netD_A.state_dict(), f"weights/{dataset_p}/netD_A_epoch_{epoch}.pth")
            torch.save(netD_B.state_dict(), f"weights/{dataset_p}/netD_B_epoch_{epoch}.pth")

            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

        # save last check pointing
        torch.save(netG_A2B.state_dict(), f"weights/{dataset_p}/netG_A2B.pth")
        torch.save(netG_B2A.state_dict(), f"weights/{dataset_p}/netG_B2A.pth")
        torch.save(netD_A.state_dict(), f"weights/{dataset_p}/netD_A.pth")
        torch.save(netD_B.state_dict(), f"weights/{dataset_p}/netD_B.pth")
        return (netG_A2B,netG_B2A,netD_A,netD_B)

    def retrieve_nets(self,dataset = "dataset",device="cpu"):
            netG_A2B = Generator().to(device)
            netG_B2A = Generator().to(device)
            netD_A = Discriminator().to(device)
            netD_B = Discriminator().to(device)

            netG_A2B.apply(weights_init)
            netG_B2A.apply(weights_init)
            netD_A.apply(weights_init)
            netD_B.apply(weights_init)

            if netG_A2B != "":
                netG_A2B.load_state_dict(torch.load(f"weights/{dataset}/netG_A2B.pth"))
            if netG_B2A != "":
                netG_B2A.load_state_dict(torch.load(f"weights/{dataset}/netG_B2A.pth"))
            if netD_A != "":
                netD_A.load_state_dict(torch.load(f"weights/{dataset}/netD_A.pth"))
            if netD_B != "":
                netD_B.load_state_dict(torch.load(f"weights/{dataset}/netD_B.pth"))
            return (netG_A2B,netG_B2A,netD_A,netD_B)
