import os

if input("Install needed packages using APT ? (y/n)").lower() == "y":
    os.system("sudo apt install python3 python3-pip")

proc = input("What is your processing unit ? (cuda/cpu) :").lower()

if proc == "cpu":
    os.system("pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu")
else:
    os.system("pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113")

os.system("pip install pillow tqdm numpy opencv-python urllib3 PyYAML")
os.system("cp templateconf.yml conf.yml")
if proc == "cpu":
    os.system("echo UseCUDA: no >> config.yml")
else:
    os.system("echo UseCUDA: yes >> config.yml")

print("You can now edit your config.yml file, manual in info.pdf")
if input("Do you want to run the train script ? (y/n)").lower() == "y":
    import train
print("Bye")
