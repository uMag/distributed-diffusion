#!/bin/bash

#Install deps
apt-get update -y
apt-get install htop screen psmisc python3-pip unzip wget gcc g++ nano -y

#Install Python deps
pip install -r requirements.txt

#optional for xformers:
conda install xformers -c xformers/label/dev

touch worked