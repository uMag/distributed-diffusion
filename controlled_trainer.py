# Install bitsandbytes:
# `nvcc --version` to get CUDA version.
# `pip install -i https://test.pypi.org/simple/ bitsandbytes-cudaXXX` to install for current CUDA.
# Example Usage:
# Single GPU: torchrun --nproc_per_node=1 trainer/diffusers_trainer.py --model="CompVis/stable-diffusion-v1-4" --run_name="liminal" --dataset="liminal-dataset" --hf_token="hf_blablabla" --bucket_side_min=64 --use_8bit_adam=True --gradient_checkpointing=True --batch_size=1 --fp16=True --image_log_steps=250 --epochs=20 --resolution=768 --use_ema=True
# Multiple GPUs: torchrun --nproc_per_node=N trainer/diffusers_trainer.py --model="CompVis/stable-diffusion-v1-4" --run_name="liminal" --dataset="liminal-dataset" --hf_token="hf_blablabla" --bucket_side_min=64 --use_8bit_adam=True --gradient_checkpointing=True --batch_size=10 --fp16=True --image_log_steps=250 --epochs=20 --resolution=768 --use_ema=True

import argparse
from asyncio import QueueEmpty
import socket
import sys
import zipfile
import torch
import torchvision
import transformers
import diffusers
import os
import glob
import random
import tqdm
import resource
import psutil
import pynvml
import wandb
import gc
import time
import itertools
import numpy as np
import json
import re
import traceback
import shutil
import requests
import hivemind
import ipaddress
from multiprocessing import Process, Queue, Manager, set_start_method
import multiprocessing
from typing import Optional
from functools import reduce
import pickle

try:
    pynvml.nvmlInit()
except pynvml.nvml.NVMLError_LibraryNotFound:
    pynvml = None

from typing import Iterable
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.optimization import get_scheduler
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageOps
from PIL.Image import Image as Img

from typing import Generator, Tuple

torch.backends.cuda.matmul.allow_tf32 = True

from omegaconf import OmegaConf
from hivemind import Float16Compression
from threading import Thread

import logging
from utils.data import Validation, Resize, ImageStore, SimpleBucket, AspectDataset, EMAModel

import omegaconf
import torch
import time

import sqlite3
import random

# Connect to the database
conn = sqlite3.connect('danbooru.db')
cursor = conn.cursor()

# Load the posts table into memory
cursor.execute('SELECT * FROM posts')
posts = cursor.fetchall()

def select_random_post():
  # Select a random record from the posts table
  random_post = random.choice(posts)
  post_id = random_post[0]
  image_ext = random_post[1]
  rating = random_post[2]
  
  # Return the post_id, image_ext, and rating
  return post_id, image_ext, rating

# Get the total number of rows in the posts table
cursor.execute('SELECT COUNT(*) FROM posts')
num_rows = cursor.fetchone()[0]

def get_num_rows():
    return num_rows

# Close the connection to the database
conn.close()

class StopTrainingException(Exception):
    pass

def get_gpu_ram() -> str:
    """
    Returns memory usage statistics for the CPU, GPU, and Torch.

    :return:
    """
    gpu_str = ""
    torch_str = ""
    try:
        cudadev = torch.cuda.current_device()
        nvml_device = pynvml.nvmlDeviceGetHandleByIndex(cudadev)
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_device)
        gpu_total = int(gpu_info.total / 1E6)
        gpu_free = int(gpu_info.free / 1E6)
        gpu_used = int(gpu_info.used / 1E6)
        gpu_str = f"GPU: (U: {gpu_used:,}mb F: {gpu_free:,}mb " \
                  f"T: {gpu_total:,}mb) "
        torch_reserved_gpu = int(torch.cuda.memory.memory_reserved() / 1E6)
        torch_reserved_max = int(torch.cuda.memory.max_memory_reserved() / 1E6)
        torch_used_gpu = int(torch.cuda.memory_allocated() / 1E6)
        torch_max_used_gpu = int(torch.cuda.max_memory_allocated() / 1E6)
        torch_str = f"TORCH: (R: {torch_reserved_gpu:,}mb/"  \
                    f"{torch_reserved_max:,}mb, " \
                    f"A: {torch_used_gpu:,}mb/{torch_max_used_gpu:,}mb)"
    except AssertionError:
        pass
    cpu_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1E3 +
                     resource.getrusage(
                         resource.RUSAGE_CHILDREN).ru_maxrss / 1E3)
    cpu_vmem = psutil.virtual_memory()
    cpu_free = int(cpu_vmem.free / 1E6)
    return f"CPU: (maxrss: {cpu_maxrss:,}mb F: {cpu_free:,}mb) " \
           f"{gpu_str}" \
           f"{torch_str}"

def setuphivemind(conf, log_queue):
    log_queue.put("Setting up hivemind")
    if os.path.exists(conf.intern.workingdir):
        shutil.rmtree(conf.intern.workingdir)
    os.makedirs(conf.intern.workingdir)

    # if requests.get('http://' + conf.server + '/info').status_code == 200:
    #     print("Connection Success")
    #     log_queue.put("Connected to the dataset server, retrieving lr_scheduler configuration")
    #     serverconfig = json.loads(requests.get('http://' + conf.server + '/v1/get/lr_schel_conf').content)
    #     print(serverconfig)
    #     imgs_per_epoch = int(serverconfig["ImagesPerEpoch"])
    #     total_epochs = int(serverconfig["Epochs"])
    #     return(imgs_per_epoch, total_epochs)
    # else:
    #     log_queue.put("Unable to connect to the dataset server")
    #     raise ConnectionError("Unable to connect to server")

import threading

def getchunk(amount, conf, log_queue):
    log_queue.put("Requesting Chunks")
    if os.path.isdir(conf.intern.tmpdataset):
        shutil.rmtree(conf.intern.tmpdataset)
    os.mkdir(conf.intern.tmpdataset)
    
    # Select 500 random records from the posts table
    random_posts = [select_random_post() for _ in range(int(amount))]
    
    threads = []
    for post_id, image_ext, rating in random_posts:
        t = threading.Thread(target=download_image, args=(post_id, image_ext, conf, log_queue))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    log_queue.put("Chunks ready")

def download_image(post_id, image_ext, conf, log_queue):
    print("Downloading some image...")
    # Download the image
    image_url = f"https://crowdcloud.us-southeast-1.linodeobjects.com/crowdcloud/opendataset/v1/danbooru/{post_id}.{image_ext}"
    image_response = requests.get(image_url)
    open(f"{conf.intern.tmpdataset}/{post_id}.{image_ext}", 'wb').write(image_response.content)

    # Download the tags
    tags_url = f"https://crowdcloud.us-southeast-1.linodeobjects.com/crowdcloud/opendataset/v1/danbooru/{post_id}.json"
    tags_response = requests.get(tags_url).json()
    tags = tags_response['tags']
    open(f"{conf.intern.tmpdataset}/{post_id}.txt", 'w').write(', '.join(tags))

def dataloader(tokenizer, text_encoder, device, world_size, rank, conf, log_queue):
    # load dataset
    log_queue.put("Setting up ImageStore")
    store = ImageStore(conf.intern.tmpdataset, conf)
    log_queue.put("Setting up AspectDataset")
    dataset = AspectDataset(store, tokenizer, text_encoder, device, conf, ucg=float(conf.everyone.ucg))
    log_queue.put("Setting up SimpleBucket")
    sampler = SimpleBucket(
            store = store,
            batch_size = int(conf.batchSize),
            shuffle = conf.everyone.buckets_shuffle,
            resize = conf.image_store_resize,
            image_side_min = int(conf.everyone.buckets_side_min),
            image_side_max = int(conf.everyone.buckets_side_max),
            image_side_divisor = 64,
            max_image_area = int(conf.everyone.resolution) ** 2,
            num_replicas = world_size,
            rank = rank
    )
    out_length = "Store Length: " + str(len(store))
    log_queue.put(str(out_length))
    print(f'STORE_LEN: {len(store)}')
    log_queue.put("Setting up Dataloader")
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )

    return train_dataloader


def InitializeTraining(command_queue, log_queue, conf):
    imgs_per_epoch = get_num_rows()
    total_epochs = 10
    config = omegaconf.OmegaConf.create(conf)
    rank = 0

    torch.cuda.set_device(rank)

    if rank == 0:
        os.makedirs(config.intern.workingdir, exist_ok=True)

        mode = 'disabled'
        if config.enable_wandb:
            mode = 'online'
        run = wandb.init(project="Hivemind Project", name="Hivemind", config=config, dir=config.intern.workingdir+'/wandb', mode=mode)

        # Inform the user of host, and various versions -- useful for debugging issues.
        print("RUN_NAME:", "Hivemind Project")
        print("HOST:", socket.gethostname())
        log_queue.put("HOST: " + str(socket.gethostname()))
        print("CUDA:", torch.version.cuda)
        log_queue.put(("CUDA: " + str(torch.version.cuda)))
        print("TORCH:", torch.__version__)
        log_queue.put(("TORCH: " + str(torch.__version__)))
        print("TRANSFORMERS:", transformers.__version__)
        log_queue.put(("TRANSFORMERS: " + str(transformers.__version__)))
        print("DIFFUSERS:", diffusers.__version__)
        log_queue.put(("DIFFUSERS:" + str(diffusers.__version__)))
        print("MODEL:", conf.everyone.model)
        log_queue.put(("MODEL:" + str(conf.everyone.model)))
        print("FP16:", conf.everyone.fp16)
        log_queue.put(("FP16:" + str(conf.everyone.fp16)))
        print("RESOLUTION:", conf.everyone.resolution)
        log_queue.put(("RESOLUTION:" + str(conf.everyone.resolution)))
    
    device = torch.device('cuda')

    print("DEVICE:", device)
    log_queue.put(("DEVICE: " + str(device)))

    # setup fp16 stuff
    scaler = torch.cuda.amp.GradScaler(enabled=conf.everyone.fp16)

    # Set seed
    torch.manual_seed(conf.everyone.seed)
    random.seed(conf.everyone.seed)
    np.random.seed(conf.everyone.seed)
    print('RANDOM SEED:', conf.everyone.seed)
    
    # I think the hf token is set to an empty string, and not None, so we should be ok. thx js
    tokenizer = CLIPTokenizer.from_pretrained(conf.everyone.model, subfolder='tokenizer', use_auth_token=conf.hftoken)
    text_encoder = CLIPTextModel.from_pretrained(conf.everyone.model, subfolder='text_encoder', use_auth_token=conf.hftoken)
    vae = AutoencoderKL.from_pretrained(conf.everyone.model, subfolder='vae', use_auth_token=conf.hftoken)
    unet = UNet2DConditionModel.from_pretrained(conf.everyone.model, subfolder='unet', use_auth_token=conf.hftoken)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    if not conf.everyone.train_text_encoder:
        text_encoder.requires_grad_(False)

    if conf.gradckpt:
        unet.enable_gradient_checkpointing()
        if conf.everyone.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if conf.xformers:
        unet.set_use_memory_efficient_attention_xformers(True)

    # "The “safer” approach would be to move the model to the device first and create the optimizer afterwards."
    weight_dtype = torch.float16 if conf.everyone.fp16 else torch.float32

    # move models to device
    vae = vae.to(device, dtype=weight_dtype)
    unet = unet.to(device, dtype=torch.float32)
    text_encoder = text_encoder.to(device, dtype=weight_dtype if not conf.everyone.train_text_encoder else torch.float32)


    if conf.eightbitadam: # Bits and bytes is only supported on certain CUDA setups, so default to regular adam if it fails.
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except:
            print('bitsandbytes not supported, using regular Adam optimizer')
            log_queue.put('bitsandbytes not supported, using regular Adam optimizer')
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    """
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )
    """

    optimizer_parameters = unet.parameters() if not conf.everyone.train_text_encoder else itertools.chain(unet.parameters(), text_encoder.parameters())

    # Create distributed optimizer
    #from torch.distributed.optim import ZeroRedundancyOptimizer
    #we changed to cls for single gpu training
    print("Stating standard optimizer")
    tmp_optimizer = optimizer_cls(
        optimizer_parameters,
        # optimizer_class=optimizer_cls,
        # parameters_as_bucket_view=True,
        lr=float(conf.everyone.lr),
        betas=(float(conf.everyone.opt_betas_one), float(conf.everyone.opt_betas_two)),
        eps=float(conf.everyone.opt_epsilon),
        weight_decay=float(conf.everyone.opt_weight_decay),
    )
    print("Finished standard optimizer")

    noise_scheduler = DDPMScheduler.from_pretrained(
        conf.everyone.model,
        subfolder='scheduler',
        use_auth_token=conf.hftoken,
    )

    # Hivemind Setup
    # get network peers (if mother peer then ignore)
    rmaddrs_rq = requests.get('http://' + conf.server + "/v1/get/peers")
    if rmaddrs_rq.status_code == 200:
        peer_list = json.loads(rmaddrs_rq.content)
    else:
        log_queue.put("Unable to obtain peers from server")
        raise ConnectionError("Unable to obtain peers from server")

    log_queue.put("Trainer set to " + conf.trainermode + " mode")
    if conf.trainermode == "Client":
        
        client_mode = True
        host_maddrs_full = None
        public_maddrs_full = None
    elif conf.trainermode == "Relay":
        client_mode = False

        # set local maddrs ports
        host_maddrs_tcp = "/ip4/0.0.0.0/tcp/" + str(conf.internal_udp)
        host_maddrs_udp = "/ip4/0.0.0.0/udp/" + str(conf.internal_tcp) + "/quic"
        host_maddrs_full = [host_maddrs_tcp, host_maddrs_udp]

        # set public to-be-announced maddrs
        # get public ip
        if conf.publicip == "":
            conf.publicip = None

        if conf.publicip == "auto" or conf.publicip is None:
            log_queue.put("Auto-detecting public IP")
            completed = False
            if completed is False:
                try:
                    ip = requests.get("https://api.ipify.org/", timeout=5).text
                    ipsrc = "online"
                    completed = True
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
                    print("Ipfy.org took too long, trying another domain.")
                    log_queue.put("Ipfy.org took too long, trying another domain.")
            if completed is False:
                try:
                    ip = requests.get("https://ipv4.icanhazip.com/", timeout=5).text
                    ipsrc = "online"
                    completed = True
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
                    print("Icanhazip.com took too long, trying another domain.")
                    log_queue.put("Icanhazip.com took too long, trying another domain.")
            if completed is False:
                try:
                    tmpjson = json.loads(requests.get("https://jsonip.com/", timeout=5).content)
                    ip = tmpjson["ip"]
                    ipsrc = "online"
                    completed = True
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
                    print("Jsonip.com took too long, ran out of alternatives.")
                    log_queue.put("Jsonip.com took too long, ran out of alternatives.")
                    raise(ConnectionError)
        else:
            log_queue.put("Loading public IP from configuration")
            ip = conf.publicip
            ipsrc = "config"
        
        #check if valid ip
        try:
            ip = ipaddress.ip_address(ip)
            ip = str(ip)
        except Exception:
            log_queue.put("Invalid IP, please check the configuration file. IP Source: " + ipsrc)
            raise ValueError("Invalid IP, please check the configuration file. IP Source: " + ipsrc)

        public_maddrs_tcp = "/ip4/" + ip + "/tcp/" + str(conf.external_tcp)
        public_maddrs_udp = "/ip4/" + ip + "/udp/" + str(conf.external_udp) + "/quic"
        public_maddrs_full = [public_maddrs_tcp, public_maddrs_udp]

    #init dht
    dht = hivemind.DHT(
        host_maddrs=host_maddrs_full,
        initial_peers=peer_list,
        start=True,
        announce_maddrs=public_maddrs_full,
        client_mode=client_mode,
    )

    #set compression and optimizer
    compression = Float16Compression()

    lr_scheduler = get_scheduler(
    conf.everyone.lr_scheduler,
    optimizer=tmp_optimizer,
    num_warmup_steps=int(float(conf.everyone.lr_scheduler_warmup) * imgs_per_epoch * total_epochs),
    num_training_steps=total_epochs * imgs_per_epoch,
    )

    print("Stating hivemind optimizer")

    optimizer = hivemind.Optimizer(
        dht=dht,
        run_id="testrun",
        batch_size_per_step=(1 * int(conf.batchSize)),
        target_batch_size=75000,
        optimizer=tmp_optimizer,
        use_local_updates=False,
        matchmaking_time=260.0,
        averaging_timeout=1200.0,
        allreduce_timeout=1200.0,
        load_state_timeout=1200.0,
        grad_compression=compression,
        state_averaging_compression=compression,
        verbose=True,
        scheduler=lr_scheduler
    )

    print("Finished hivemind optimizer")

    print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
    print("Global IP:", hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs()))
    log_queue.put("Hivemind Optimizer and DHT started successfully!")
    log_queue.put("You can share the following initial_perrs to other nodes so they connect directly through this node:")
    log_queue.put('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
    log_queue.put("Global IP:", hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs()))

    #statistics
    if conf.enablestats:
        statconfig = {"geoaprox": False, "bandwidth": False, "specs": False}
        bandwidthstats = {}
        specs_stats = {}
        print("Stats enabled")
        log_queue.put("Public Telemetry enabled.")

        if conf.geoaprox:
            log_queue.put("Geolocation Aproximation enabled (server-side)")
            statconfig['geoaprox'] = True

        if conf.bandwidth:
            log_queue.put("Bandwidth enabled (client-side)")
            statconfig["bandwidth"] = True
            import speedtest
            session = speedtest.Speedtest()
            download = session.download()
            upload = session.upload()
            ping = session.results.ping
            bandwidthstats = {"download": str(download), "upload": str(upload), "ping": str(ping)}

        if conf.specs:
            log_queue.put("Specs enabled (client-side)")
            statconfig["specs"] = True
            # GPU
            # https://docs.nvidia.com/deploy/nvml-api/index.html
            pynvml.nvmlInit()
            cudadriver_version = pynvml.nvmlSystemGetCudaDriverVersion()
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            NVML_version = pynvml.nvmlSystemGetNVMLVersion()

            #TODO: Assuming one gpu only
            cudadev = torch.cuda.current_device()
            nvml_device = pynvml.nvmlDeviceGetHandleByIndex(cudadev)

            #psu_info = pynvml.nvmlUnitGetPsuInfo(pynvml.c_nvmlPSUInfo_t.)
            #temperature_info = pynvml.nvmlUnitGetTemperature(nvml_device)
            #unit_info = pynvml.nvmlUnitGetUnitInfo(nvml_device)

            arch_info = pynvml.nvmlDeviceGetArchitecture(nvml_device)
            brand_info = pynvml.nvmlDeviceGetBrand(nvml_device)
            #clock_info = pynvml.nvmlDeviceGetClock(nvml_device)
            #clockinfo_info = pynvml.nvmlDeviceGetClockInfo(nvml_device)
            #maxclock_info = pynvml.nvmlDeviceGetMaxClockInfo(nvml_device)
            computemode_info = pynvml.nvmlDeviceGetComputeMode(nvml_device)
            compute_compatability = pynvml.nvmlDeviceGetCudaComputeCapability(nvml_device)

            pcie_link_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(nvml_device)
            pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(nvml_device)

            display_active_bool = pynvml.nvmlDeviceGetDisplayActive(nvml_device)

            #memory_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_device)

            gpu_energy_cons = pynvml.nvmlDeviceGetTotalEnergyConsumption(nvml_device)
            device_name = pynvml.nvmlDeviceGetName(nvml_device)

            gpusinfo = {
                "software": {
                    "CUDA_DRIVER_VERSION": str(cudadriver_version),
                    "NVIDIA_DRIVER_VERSION": str(driver_version),
                    "NVML_VERSION": str(NVML_version),
                },
                "hardware": {
                    "energy": {
                        #"PSU_INFO": psu_info,
                        #"TEMPERATURE_INFO": temperature_info,
                        "ENERGY_CONSUMPTION": str(gpu_energy_cons)
                    },
                    "info": {
                        #"UNIT_INFO": unit_info,
                        "BRAND_INFO": str(brand_info),
                        "DEV_NAME": str(device_name),
                        "DISPLAY_ACTIVE": str(display_active_bool),
                        "ARCH_INFO": str(arch_info)
                    },
                    "memory": {
                        "PCIE_LINK_GEN": str(pcie_link_gen),
                        "PCIE_WIDTH": str(pcie_width),
                        #"MEMORY_INFO": memory_info,
                    },
                    "compute": {
                        #"CLOCK": clock_info,
                        #"CLOCK_INFO": clockinfo_info,
                        #"MAX_CLOCK": maxclock_info,
                        "COMPUTE_MODE": str(computemode_info),
                        "COMPUTE_COMPATABILITY": str(compute_compatability)
                    }
                }
            }

            cpuinfo = {}
            import cpuinfo
            cpudict = cpuinfo.get_cpu_info()
            cpuinfo = {
                'CPU_ARCH': str(cpudict['arch']),
                "CPU_HZ_AD": str(cpudict["hz_advertised_friendly"]),
                "CPU_HZ_AC": str(cpudict["hz_actual_friendly"]),
                "CPU_BITS": str(cpudict["bits"]),
                "VENDOR_ID": str(cpudict["vendor_id_raw"]),
                #"HARDWARE_RAW": cpudict["hardware_raw"],
                "BRAND_RAW": str(cpudict["brand_raw"])
            }

            specs_stats = {'gpu': gpusinfo, 'cpu': cpuinfo}
        statsjson = {
            'python_ver': str(sys.version),
            'config': statconfig,
            'bandwidth': bandwidthstats,
            'specs': specs_stats
        }
        print(statsjson)
        pstats = requests.post('http://' + conf.server + '/v1/post/stats', json=json.dumps(statsjson))
        if pstats.status_code != 200:
            log_queue.put("Failed to report telemetry")
            raise ConnectionError("Failed to report stats")
        else:
            log_queue.put("Telemetry reported successfully")
   
    # create ema
    if conf.everyone.use_ema:
        ema_unet = EMAModel(unet.parameters())

    print(get_gpu_ram())

    def save_checkpoint(global_step):
        if rank == 0:
            if conf.everyone.use_ema:
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())
            pipeline = StableDiffusionPipeline(
                text_encoder=text_encoder, #if type(text_encoder) is not torch.nn.parallel.DistributedDataParallel else text_encoder.module,
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
                scheduler=PNDMScheduler.from_pretrained(conf.everyone.model, subfolder="scheduler", use_auth_token=conf.hftoken),
                safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
            )
            log_queue.put(f'saving checkpoint to: {conf.intern.workingdir}/{"hivemind"}_{global_step}')
            print(f'saving checkpoint to: {conf.intern.workingdir}/{"hivemind"}_{global_step}')
            pipeline.save_pretrained(f'{conf.intern.workingdir}/{"hivemind"}_{global_step}')

            if conf.everyone.use_ema:
                ema_unet.restore(unet.parameters())
    
    # train!
    try:
        already_done_steps = (optimizer.tracker.global_progress.samples_accumulated + (optimizer.tracker.global_progress.epoch * optimizer.target_batch_size))
        print("Skipping", already_done_steps, "steps on the LR Scheduler.")
        log_queue.put("Skipping " + str(already_done_steps) + " steps on the LR Scheduler.")
        for i in range(already_done_steps):
            lr_scheduler.step()
        print("Done")
        loss = torch.tensor(0.0, device=device, dtype=weight_dtype)
        global_step = 0
        while True:
            print(get_gpu_ram())
            #only provide domain (ex.: 127.0.0.1:8080 or sail.pe:9000) here, http:// is added in the function.
            getchunk(conf.imageCount, conf, log_queue)

            #Note: we removed worldsize here
            train_dataloader = dataloader(tokenizer, text_encoder, device, 1, rank, conf, log_queue)
            num_steps_per_epoch = len(train_dataloader)
            progress_bar = tqdm.tqdm(range(num_steps_per_epoch), desc="Total Steps", leave=False)
            
            unet.train()
            if conf.everyone.train_text_encoder:
                text_encoder.train()

            for _, batch in enumerate(train_dataloader):
                if command_queue.qsize() > 0:
                    command = command_queue.get()
                    if command == 'stop':
                        # Start training
                        print('Stopping training...')
                        log_queue.put("Stopping training...")
                        raise StopTrainingException("Recieved Stop Training Command.")
                
                b_start = time.perf_counter()
                latents = vae.encode(batch['pixel_values'].to(device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the embedding for conditioning
                encoder_hidden_states = batch['input_ids']

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                with torch.autocast('cuda', enabled=conf.everyone.fp16):
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                # backprop and update
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                if conf.everyone.train_text_encoder:
                    torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Update EMA
                if conf.everyone.use_ema:
                    ema_unet.step(unet.parameters())

                # perf
                b_end = time.perf_counter()
                seconds_per_step = b_end - b_start
                steps_per_second = 1 / seconds_per_step
                rank_images_per_second = int(conf.batchSize) * steps_per_second
                #world_images_per_second = rank_images_per_second #* world_size
                samples_seen = global_step * int(conf.batchSize) #* world_size

                # get global loss for logging
                # torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                loss = loss #/ world_size

                if rank == 0:
                    progress_bar.update(1)
                    global_step += 1
                    logs = {
                        "train/loss": loss.detach().item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": 1,
                        "train/step": global_step,
                        "train/samples_seen": samples_seen,
                        "perf/rank_samples_per_second": rank_images_per_second,
                        #"perf/global_samples_per_second": world_images_per_second,
                    }
                    progress_bar.set_postfix(logs)
                    run.log(logs, step=global_step)
                    if global_step % 10 == 0 and global_step > 0:
                        first_str_to_log = "steps/s: " + str(steps_per_second) + " imgs/s: " + str(rank_images_per_second) + " imgs seen: " + str(samples_seen)
                        second_str_to_log = "loss: " + str(loss.detach().item()) + " lr: " + str(lr_scheduler.get_last_lr()[0]) + " step: " + str(global_step)
                        first_opt_log = str(optimizer.tracker.global_progress)
                        log_queue.put(first_str_to_log)
                        log_queue.put(second_str_to_log)
                        log_queue.put(first_opt_log)
                    #tqdm_out = progress_bar.format_dict()
                    #print(str(tqdm_out))
                    # if counter < 5:
                    #     counter += 1
                    # elif counter >= 5:
                    #     data = {
                    #         "tracker.global_progress": optimizer.tracker.global_progress,
                    #         "tracker.local_progress": optimizer.tracker.local_progress,
                    #     }
                    #     print(data)
                    #     counter = 0
                    #Thread(target=backgroundreport, args=(("http://" + conf.server + "/v1/post/ping"), "world_images_per_second")).start()

                if global_step % int(conf.savesteps) == 0 and global_step > 0:
                    save_checkpoint(global_step)
                    
                if conf.enable_inference:
                    #hardcoded
                    if global_step % 500 == 0 and global_step > 0:
                        if rank == 0:
                            # get prompt from random batch
                            prompt = tokenizer.decode(batch['tokens'][random.randint(0, len(batch['tokens'])-1)])

                            if conf.image_inference_scheduler == 'DDIMScheduler':
                                print('using DDIMScheduler scheduler')
                                scheduler = DDIMScheduler.from_pretrained(conf.everyone.model, subfolder="scheduler", use_auth_token=conf.hftoken)
                            else:
                                print('using PNDMScheduler scheduler')
                                scheduler=PNDMScheduler.from_pretrained(conf.everyone.model, subfolder="scheduler", use_auth_token=conf.hftoken)

                            pipeline = StableDiffusionPipeline(
                                text_encoder=text_encoder, #if type(text_encoder) is not torch.nn.parallel.DistributedDataParallel else text_encoder.module,
                                vae=vae,
                                unet=unet.module,
                                tokenizer=tokenizer,
                                scheduler=scheduler,
                                safety_checker=None, # disable safety checker to save memory
                                feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
                            ).to(device)
                            # inference
                            if conf.enable_wandb:
                                images = []
                            else:
                                saveInferencePath = conf.intern.workingdir + "/inference"
                                os.makedirs(saveInferencePath, exist_ok=True)
                            with torch.no_grad():
                                with torch.autocast('cuda', enabled=conf.everyone.fp16):
                                    #hardcoded, twice
                                    for _ in range(5):
                                        if conf.local.wandb:
                                            images.append(
                                                wandb.Image(pipeline(
                                                    prompt, num_inference_steps=30
                                                ).images[0],
                                                caption=prompt)
                                            )
                                        else:
                                            from datetime import datetime
                                            #hardcoded
                                            images = pipeline(prompt, num_inference_steps=30).images[0]
                                            filenameImg = str(time.time_ns()) + ".png"
                                            filenameTxt = str(time.time_ns()) + ".txt"
                                            images.save(saveInferencePath + "/" + filenameImg)
                                            with open(saveInferencePath + "/" + filenameTxt, 'a') as f:
                                                f.write('Used prompt: ' + prompt + '\n')
                                                f.write('Generated Image Filename: ' + filenameImg + '\n')
                                                f.write('Generated at: ' + str(global_step) + ' steps' + '\n')
                                                f.write('Generated at: ' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+ '\n')

                            # log images under single caption
                            if conf.enable_wandb:
                                run.log({'images': images}, step=global_step)

                            # cleanup so we don't run out of memory
                            del pipeline
                            gc.collect()
    except StopTrainingException as e:
        print("Stopping Training upon user request")
        print("TRAINING_STOPPED")
        log_queue.put("TRAINING STOPPED")
        pass
    except Exception as e:
        print(f'Exception caught on rank {rank} at step {global_step}, saving checkpoint...\n{e}\n{traceback.format_exc()}')
        pass

    save_checkpoint(global_step)

    #cleanup()

    print(get_gpu_ram())
    print('Done!')
    print("TRAINING_FINISHED")
    log_queue.put("TRAINING FINISHED")
    exit()

def PyTorchTrainer(command_queue, log_queue):
    print(type(command_queue), flush=True)
    print(command_queue, flush=True)
    while True:
        command = command_queue.get()
        if command == 'start':
            print('Starting Training!')
            with open("DO_NOT_DELETE_config.pickle", 'rb') as f:
                    conf = pickle.load(f)
            #stop is gonna be done inside the function, must change this later
            InitializeTraining(command_queue, log_queue, conf)
        elif command == 'stop':
            #kill before it even starts????
            print('Bye!')
            return
