# Distributed Diffusion
Train Stable Diffusion models across the internet with multiple peers

This project is in alpha and is under testing


## Requirements

8 Bit Adam:
- 17.4GB of VRAM
- 12.5GB of RAM

8 Bit Adam and FP16 (BROKEN):
- 19GB of VRAM
- 8GB of RAM

- A good and stable internet connection
- Python 3
- Linux or WSL (due to UVLOOP)
- Have installed dependencies on requirements.txt

## How to use

Ex.: 
```
python3 finetune.py --dataset ./directory_of_my_dataset --model ./diffuser_folder_of_base_model --run_name my_finetuned_model
```

### A step-by-step guide is available here: https://rentry.org/informal-training-guide

There is a long list of flags you can set on the script, but the most important ones are the following:

Required flags:
- --model : Set the path to model, must be in diffusers form
- --run_name : Name for the training run, does not interfere with Hivemind
- --dataset : Set the path to the dataset, images and text files must be in the same folder

Optional Flags:
- --lr : Set the Lreaning rate, float type, default is 5e-6.
- --epochs: Set the number of epochs ([Definition](https://deepai.org/machine-learning-glossary-and-terms/epoch)) to train for, integer type, deafult is 10.
- --batch_size : Batch size to use. Fills more VRAM but trains faster. integer type, default is 1.
- --center_crop : Crop Images during training to it's center. Default is True, if disabled it might break if it gets a 512x512+ image.
- --fp16 : BROKEN. Train in mixed precision, accelerating the process. Default is False, and according to some users it is currently broken since loss gets to NaN in the long run.

Hivemind Specific:
- --peers : Set the peers to connect to, separated by space bar and as a list. If None then a new session will be started. Ex.: --peers /ipv4/1.1.1.1 /ipv4/2.2.2.2
- --hivemind: Enable hivemind distributed training (ALPHA)
- --hfstream : Stream dataset from HuggingFace.

More flags: https://github.com/chavinlo/distributed-diffusion/blob/main/finetune.py#L34

## Dataset format
At the moment there are two forms of loading a dataset:
### Directory based:
Put all the images and texts into one directory. Should look like the following:

![image](https://user-images.githubusercontent.com/85657083/198753014-bb8947d4-618a-4f85-a480-197c141a6fbf.png)

Accepted image extensions: PNG, JPG, JPEG, BMP, WEBP

As you can see, both the images and text files must be in the same folder, and have the same name to link each other.
The Image file must contain the image itself, and the text file must contain the text to correlate to. These two must have the same filename to identify and match them.

For example:
Image file: 5701024.jpg

Contents:

<img src="https://user-images.githubusercontent.com/85657083/198753223-dcd059c1-6e20-4028-9fe0-387cc4890449.png" width="360" height="360" />

Source: @Peargor on Twitter, https://twitter.com/Peargor/status/1574190864267608071

Text file: 5701024.txt

Contents:

```
1koma, 2girls, black eyes, brown hair, chair, comic, computer, curtains, english text, feet out of frame, indoors, laptop, looking at another, multiple girls, office chair, open mouth, red skirt, school uniform, shirt, short hair, short twintails, sitting, skirt, smile, table, thighhighs, tokisadame school uniform, twintails, white thighhighs, yellow shirt, aioi yuuko, naganohara mio, absurdres, commentary, highres, peargor, nichijou
```

Once you have your dataset directory ready, add the "--dataset" flag to your command, followed by a space and the path to your directory.

Ex.: 
```
python3 finetune.py --dataset ./directory_of_my_dataset --model ./sd-diffuser --run_name my_finetuned_model
```

### HuggingFace Dataset Streaming (Recommended with hivemind):
Note: I haven't tested this yet so it might not work

This method consists on using HuggingFace's Dataset module to stream the data from their site eliminating the requirement to download the entire dataset. This can be useful specially when using Hivemind in order to distribute the data evenly, as seen on Hivemind's Training-Transformers-Together experiment: https://training-transformers-together.github.io/#memory-efficiency

For more information please check HuggingFace's docs:
https://huggingface.co/docs/datasets/how_to


## How to join
We are going to start a collaborative training run around november.

If you want to get notified, join this server: https://discord.gg/8Sh2T6gjd2

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
- Offload to NVMe with zero-infinity? https://github.com/huggingface/transformers/issues/11464
- Syncronize dataset automatically
- Test aronund dataset progress and its effects on loss
- Compress even further the syncronization
- Live global progress site
- Aproximation to achieve same compute as 8 A100s

## Statistics
Experiments ran on A100-80GB-XM4
ETA is for 75K Images/10 Epochs/750K Itterations.
Loaded via directories

8 Bit Adam only, Batch size 1, lr 5e-6:
- It/s: Min. 2.60; Max. 3.20; Avg. 3.05
- ETA: Around 70 Hours
- 17.4GB of VRAM
- 12.5GB of RAM

8 Bit Adam only, Batch size 4, lr 5e-6:
- It/s: Min. 1.02; Max. 1.26; Avg. 1.15
- ETA: Around 47 Hours
- 37GB of VRAM
- 11.2GB of RAM

8 Bit Adam only, Batch size 8, lr 5e-6:
- s/it: Min. 1.39; Max. 2.16; Avg. 1.40
- ETA: Around 40 Hours
- 63GB of VRAM
- 11.2GB of RAM

## Credits:
- Haru: Wrote the first trainer and rewrote it from scratch optimizing it
- dep (me): Hivemind integration
