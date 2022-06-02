from tqdm import tqdm
import urllib.request
import cv2

url="https://picsum.photos/200/300"

for i in tqdm(range(0,50)):
    urllib.request.urlretrieve(url, "dataset/train/A/"+str(i)+".png")
    originalImage = cv2.imread("dataset/train/A/"+str(i)+".png")
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    backtorgb = cv2.cvtColor(grayImage,cv2.COLOR_GRAY2RGB)
    cv2.imwrite("dataset/train/B/"+str(i)+".png", backtorgb)
