:begin
@echo off
set /p proc="Enter your processing unit (cpu / cuda), cuda default: "

copy templateconf.yml config.yml

set proc="%proc%"

if %proc% == "cpu" (
	goto dlcpu
)
if %proc% == "CPU" (
	goto dlcpu
)
if %proc% == "cuda" (
	goto dlcuda
)
if %proc% == "CUDA" (
	goto dlcuda
)

goto dlcuda

:dlcuda
echo UseCUDA: yes >> config.yml
py -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
echo Downloading pytorch cuda
goto end

:dlcpu
echo UseCUDA: no >> config.yml
echo Downloading pytorch cpu
py -m pip install torch torchvision torchaudio
goto end

:end

echo Downloading and converting Images
py -m pip install pillow tqdm numpy opencv-python urllib3 PyYAML
cd data
py download.py
cd ..
echo "You can now run WindowsTrain.bat, to train the model"
:ext
echo Thank you