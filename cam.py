import numpy as np
import cv2
import torchvision.utils as vutils
import torchvision.transforms as transforms
import libcygtrn
from libcygtrn import CYCLES
from PIL import Image

image_size = 256


cyc = CYCLES()

netG_A2B,netG_B2A,netD_A,netD_B = cyc.retrieve_nets()


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

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # Display the resulting frame
    cv2.imshow('Viewfinder, press q to exit and c to process', gray)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('c'):
        nimg = pre_process(Image.fromarray(gray)).unsqueeze(0)
        calc = netG_B2A(nimg.to(device)).detach().to("cpu")
        vutils.save_image(calc, "result.png", normalize=True)
        calc = cv2.imread("result.png")
        cv2.imshow('Processed, press any key to exit',calc)
        cv2.waitKey(0)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
