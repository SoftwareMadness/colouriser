@echo off

echo "Hello, starting to train the model. If you dont have infinite time feel free to edit the 'ep' value in the config.yml  instructions below"
echo .
color F0
echo .
echo "Open config.yml"
echo .
echo "It should look something like this"
echo .
echo "UseCUDA: yes"
echo .
echo "ep: 100"
echo .
echo "e.t.c."
echo .
echo "Change ep: 100 to ep: 50            instead of 50 you can use any number you like, it is the number of epochs, more will tabe more time but will look better"
echo .
color 02
pause
color
py train.py