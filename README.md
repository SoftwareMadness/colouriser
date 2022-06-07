# <span style="color:green">Colouriser</span>
My AI Project to colorise images, it is based on the awesome pytorch-cyclegan project https://github.com/Lornatang/CycleGAN-PyTorch
<br>
<br>

State Changes:<br>
<br>  I changed the train.py file in the pytorch-cyclegan repository to a lightly stripped down version. It is still the same GAN but i can use python fucntions.
<br>  I created some new scripts with a very primitive cli. I trained the GAN to 43 epochs. It can turn Black/White to Color or in reverse.

<br>
<br>
<br>
If you want to change the dataset size change:
<br>
<br>
```for i in tqdm(range(0,500)):``` <--- Change the 500 to the number of images you want in data/download.py
<br>
Here is a image of the result. It is 13 epochs of training.
<br>

![Output](result.png)
<=== 
![Input](89.png)
