# Distributed Diffusion
Train Stable Diffusion models across the internet with multiple peers

This project is in alpha and is under testing


## Requirements

- 22GB of VRAM
- 8GB of RAM
- A good and stable internet connection
- Python 3
- Linux or WSL (due to UVLOOP)
- Have installed dependencies on requirements.txt

## How to use

There is a long list of flags you can set on the script, but the most important ones are the following:

- --model : Set the path to model, must be in diffusers form
- --run_name : Name for the training run, does not interfere with Hivemind
- --dataset : Set the path to the dataset, images and text files must be in the same folder
- --peers : Set the peers to connect to, separated by space bar and as a list. If None then a new session will be started. Ex.: --peers /ipv4/1.1.1.1 /ipv4/2.2.2.2

## How to join
We are going to start a collaborative training run around november.

If you want to get notified, join this server: [soon]

And grab the "relay" role.

## How it works
You can read more about Hivemind and its functionality on its website: [Hivemind ReadTheDocs](https://learning-at-home.readthedocs.io/en/latest/index.html)

In short terms, this is what the network does:
1. Peers join together and start a hive (collaborative) training session
2. Parameters and Gradients are syncronized
3. They train for a while
4. Once the target batch size is reached globally, all peers will exchange parameters resulting in a loss decrease
5. If a peer disconnects, even if its the main peer, the rest can continue training.

## Future Aims/Contributions
The following is a list of what we plan to do next:
- Test FP16 and shrink VRAM usage even lower
- Syncronize dataset automatically
- Test aronund dataset progress and its effects on loss
- Compress even further the syncronization
- Live global progress site
- Aproximation to achieve same compute as 8 A100s


## Credits:
- Haru: Wrote the first trainer and rewrote it from scratch optimizing it
- dep (me): Hivemind integration