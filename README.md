# Distributed Diffusion
Train Stable Diffusion models across the internet with multiple peers

### Current Version: 0.1a - First Contact
#### Release Notes
 - Dataset Server:
 - - Initial Peer Address Exchange functional
 - - Dataset Chunks working
 - - Chunk retrival, reporting, and timeout working
 - Trainer:
 - - Dataset Syncronization fully functional
 - - Dataset Reloading fully functional
 - - Option for instances with port forwarding (such as docker or NATs) added
 - - Gradient Syncronization working (with FP16 and BlockwiseQuantization)
 - - LR scheduler disabled

## How it works
Workflow:
![img](https://i.imgur.com/620cU8K.png)

Connection Map:
![img](https://i.imgur.com/QvTajeV.png)

## Usage
Note: Currently the software has many issues, and has barely any security measures against attackers. It is highly recommended to use this only between private parties.

Clone the repository

### Starting the dataset server
Remember, that the dataset server must have a large bandwidth capacity, cpu resources, and disk space

Put your dataset (txt and img files) in a folder. Let's call this folder "datasetfolder".

Install dependencies:

`pip install flask keyboard zipfile`

Next, start the server:

`python3 utils/dataset/online/server/main.py --dataset="datasetfolder" --name="Awesome Dataset" --description="Welcome to my server" --tasktimeout 25`

Where:

datasetfolder is the folder where the dataset files are located

"Awesome Dataset" the server name

"Welcome to my server" the server description

25 being the time in minutes in which a node must report an epoch completition or else his task will be reassigned to another node

This will start a server on port 8080. You can change the port at the last line of the script (utils/dataset/online/server/main.py).

### Starting the Main Node
The first node MUST announce its maddrs, so it must have 2 ports open:

A TCP port, and a UDP port. It is known that the TCP port is much more reliable than the UDP port.

On the Instance, clone this repo

Make setup.sh executable: `chmod +x setup.sh`

And execute it: `./setup.sh`

This will install the required dependencies

Once done, make run.sh executable: `chmod +x run.sh`

Now we will start configuring the startup, this is very important so be careful:

Open run.sh either via nano: `nano run.sh` or on a text editor.

You will get the following:
```
torchrun --nproc_per_node=1 \
	train.py \
	--workingdirectory hivemindtemp \
	--wantedimages 500 \
	--datasetserver="DATASET_SERVER_IP" \
	--node="true" \
	--o_port1=LOCAL_TCP_PORT \
	--o_port2=LOCAL_UDP_PORT \
	--ip_is_different="true" \
	--p_ip="PUBLIC_IP" \
	--p_port1=PUBLIC_TCP_PORT \
	--p_port2=PUBLIC_UDP_PORT \
	--batch_size 2 \
	--use_xformers="true" \
	--save_steps 1000 \
	--image_log_steps 400 \
	--hf_token="YOUR HUGGIGNFACE TOKEN" \
	--model runwayml/stable-diffusion-v1-5 \
	--run_name testrun1 \
	--gradient_checkpointing="true" \
	--use_8bit_adam="false" \
	--fp16="true" \
	--resize="true" \
	--wandb="false" \
	--no_migration="true" \
```
It should be self-explanatory what you should do here, but if you need help:

hivemindtemp is the folder where the trainer will work in

500 is the number of images per chunk

DATASET_SERVER_IP must be replaced by the ip of the dataset server (with ip:port, like 152.102.62.59:8080)

the `true` next to node should be changed to false, as this will not be a node, so it won't have initial peers.

the 2 next to batch_size is the batchsize. On RTX3090s with Xformers, the max is 2. On RTX3090s WITHOUT Xformers, the max is 1.

the `true` next to use_xformers is to enable xformers. If during the setup.sh, you got an error related to conda or xformers being unable to get installed, set this to `false`.

the 1000 next to save_steps is the amount of steps local it will save the model at. If 1000, it will save the model at 1000 steps, 2000 steps, 3000 steps, and so on.

the 400 next to image_log_steps is the amount of steps it will generate samples at, with a random prompt from the dataset provided. Same working as save_steps.

YOUR HUGGINGFACE TOKEN should be replaced with a huggingface token if using a huggingface model

runwayml/stable-diffusion-v1-5 should be replaced with the model you are going to train on top of.

the rest should be left like they are.


#### Networking configuration
LOCAL_TCP_PORT and LOCAL_UDP_PORT must be changed to the local ports respectively.

On many systems, hivemind will not report the correct public IP. Or maybe you could be running this on a docker or firewalled instance. If this is the case:

set ip_is_different to true

change PUBLIC_IP to your public IP

PUBLIC_TCP_PORT and PUBLIC_UDP_PORT to the public ports pointing to the local ports, like:

PUBLIC_TCP_PORT ---points to---> LOCAL_TCP_PORT

PUBLIC_UDP_PORT ---points to---> LOCAL_UDP_PORT

If you are 100% sure hivemind will report the correct public IP, set ip_is_different to false, and PUBLIC_TCP_PORT and PUBLIC_UDP_PORT to some random number (like 0, just a number so it doesn't argues)

### Starting more Nodes

Use the same configurations as above, but set `true` next to node. 

## Contributing & Support

All contributions are welcome! Currently I'm the only developer in this project so I would be happy for more people to join!

Soon I will set up a patreon or some way to get donations, as testing this is very expensive. 

If you need help, or want to know more about the project, join the discord: https://discord.gg/8Sh2T6gjd2

## Issues & Bugs:

- Reporting DHT maddrs from peers in a VERY incorrect manner, prevents a node from the same ip closing and reconnecting.
- Too many dead DHT maddrs from peers prevent proper initialization of the DHT client. Add a ping module from the dataset server
- Memory leak requiring to run `killall python` everytime the trainer is closed
- No security whatsoever

## Credits:
- Haru: Wrote the first trainer and rewrote it from scratch optimizing it
- dep (me): Hivemind integration, dataset server
