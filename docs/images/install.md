# Installation

Note: instructions might be inprecise because I wrote this in 5 minutes

If you are on windows, go download WSL: https://learn.microsoft.com/en-us/windows/wsl/install and chose ubuntu

1.- Clone the repo:

`https://github.com/chavinlo/distributed-diffusion`

2.- Go into it:

`cd distributed-diffusion`

3.- Run the following:

```
apt update
apt install wget screen htop psmisc git
python3 -m venv env
source env/bin/activate
python -m pip install pip
python -m pip install setuptools
python -m pip install wheel ninja
python -m pip install torch torchvision --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu117
# xformers extra deps
wget https://download.pytorch.org/whl/nightly/torchtriton-2.0.0%2Bf16138d447-cp310-cp310-linux_x86_64.whl
pip install torchtriton-2.0.0+f16138d447-cp310-cp310-linux_x86_64.whl
python -m pip install networkx
# xformers
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
cd ..
CUDA_HOME=/usr/local/cuda python -m pip install --force-reinstall -e xformers
# bitsandbytes
python -m pip install bitsandbytes
python -m pip install -r requirements.txt
```

4.- Start the server:

`python3 server.py`

4.- Docker container alternative
for windows run
```
docker run -it -v $PWD/workplace:/distributed-training/workplace -v $PWD/huggingface:/root/.cache/huggingface --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 5080:5080 ghcr.io/chavinlo/distributed-diffusion:latest
```
worplace dir - is where images will be temporaly stored and downloaded as well as intermediate wights,
huggingface dir - is where base models from hf will be cached and stored between launches of docker image
--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 - magick line to solve some memory issues on win,
-p 5080:5080 - port mapping for webui 

for linux
```
docker run -it -v ./workplace:/distributed-training/workplace -v ./huggingface:/root/.cache/huggingface --gpus=all -p 5080:5080 ghcr.io/chavinlo/distributed-diffusion:latest
```

you can also have your full path as wellfor workplace and hf cache

An address will pop up, thats the web ui.

additionaly, if you don't have access to the instance network, you can add `-t` to get a cloudflare tunnel link

5.- Fill the config like this:

![image](https://user-images.githubusercontent.com/85657083/210300996-cd5a0774-6fc2-445c-8bba-6d27fa9182cc.png)

Try to set a lower batch size to avoid out of memory

Statistics are optional, but helps us decorate the world map: https://stats.sail.pe/

If you wish to expand the DHT network, you can set the server mode to RELAY:

![image](https://user-images.githubusercontent.com/85657083/210301030-0641b83c-3404-4e5d-9376-44a20fae02ff.png)

Where:

- auto means automatic ip prediction
- tcp ports are for the dht
- IE ports are for the webserver (config)
- Internal ports are ports internal to the instance
- External ports are ports publicly accessible to the internet

All ports are TCP. None are UDP.

6.- Go to the home menu and press start 

![image](https://user-images.githubusercontent.com/85657083/210301140-7ad4e8a2-1a7b-4ece-97be-3ef8f690e00a.png)

It will take a while to download the required files, and depending on the amount of chunk images you set (500 is ok)

To save the current state of the model, press Save

To stop the training, press Stop.


Bye bye

