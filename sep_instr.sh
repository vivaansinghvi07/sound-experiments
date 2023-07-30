#!/bin/bash
echo Acquiring files...
wget https://github.com/tsurumeso/vocal-remover/releases/download/v5.0.4/vocal-remover-v5.0.4.zip &> /dev/null
unzip vocal-remover*.zip &> /dev/null
rm vocal-remover*.zip

echo Getting Instrumental ... 
cd vocal-remover
pip3 install -r requirements.txt &>/dev/null
pip3 install torch resampy &>/dev/null
python3 inference.py -i ../$1.wav -B 8
mv "$1"_Instruments.wav ../songs/"$1"_instr.wav
cd ..
rm -rf vocal-remover/

echo Instrumental saved.