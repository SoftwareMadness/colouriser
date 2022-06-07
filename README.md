# colouriser
My AI Project to colorise images, it is based on the awesome pytorch-cyclegan project https://github.com/Lornatang/CycleGAN-PyTorch


State Changes:
  I changed the train.py file in the pytorch-cyclegan repository to a lightly stripped down version. It is still the same GAN but i can use python fucntions.
  I created some new scripts with a very primitive cli. I trained the GAN to 43 epochs. It can turn Black/White to Color or in reverse.



If you want to change the dataset size change:

for i in tqdm(range(0,500)): <--- Change the 500 to the number of images you want in data/download.py

Here is a image of the result. It is 13 epochs of training.

![Output](result.png) <=== ![Input](89.png)
