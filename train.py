#python3 train.py -c CONFIGURATION.yaml

import argparse
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
from typing import Optional
from functools import reduce

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

parser = argparse.ArgumentParser(description="Hivemind Trainer")
parser.add_argument('-c', '--config', type=str, default="configuration.yaml", required=True, help="Path to the configuration YAML file")
#TODO: change this to integers
parser.add_argument('-l', '--loglevel', type=str, default="INFO", help="Loglvel for logging. https://docs.python.org/3/library/logging.html")
args = parser.parse_args()

logging.basicConfig(level="INFO")

conf = OmegaConf.load(args.config)
temporary_dataset = os.path.join(conf.local.working_path, "dataset")
cookies = {
    'nickname': conf.local.iden.nickname
}

# def setup():
#     torch.distributed.init_process_group("nccl", init_method="env://")

# def cleanup():
#     torch.distributed.destroy_process_group()

# def get_rank() -> int:
#     if not torch.distributed.is_initialized():
#         return 0
#     return torch.distributed.get_rank()

# def get_world_size() -> int:
#     if not torch.distributed.is_initialized():
#         return 1
#     return torch.distributed.get_world_size()

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

class Validation():
    def __init__(self, is_skipped: bool, is_extended: bool) -> None:
        if is_skipped:
            self.validate = self.__no_op
            return print("Validation: Skipped")

        if is_extended:
            self.validate = self.__extended_validate
            return print("Validation: Extended")

        self.validate = self.__validate
        print("Validation: Standard")

    def __validate(self, fp: str) -> bool:
        try:
            Image.open(fp)
            return True
        except:
            print(f'WARNING: Image cannot be opened: {fp}')
            return False

    def __extended_validate(self, fp: str) -> bool:
        try:
            Image.open(fp).load()
            return True
        except (OSError) as error:
            if 'truncated' in str(error):
                print(f'WARNING: Image truncated: {error}')
                return False
            print(f'WARNING: Image cannot be opened: {error}')
            return False
        except:
            print(f'WARNING: Image cannot be opened: {error}')
            return False

    def __no_op(self, fp: str) -> bool:
        return True

class Resize():
    def __init__(self, is_resizing: bool, is_not_migrating: bool) -> None:
        if not is_resizing:
            self.resize = self.__no_op
            return

        if not is_not_migrating:
            self.resize = self.__migration
            dataset_path = os.path.split(temporary_dataset)
            self.__directory = os.path.join(
                dataset_path[0],
                f'{dataset_path[1]}_cropped'
            )
            os.makedirs(self.__directory, exist_ok=True)
            return print(f"Resizing: Performing migration to '{self.__directory}'.")

        self.resize = self.__no_migration

    def __no_migration(self, image_path: str, w: int, h: int) -> Img:
        return ImageOps.fit(
                Image.open(image_path),
                (w, h),
                bleed=0.0,
                centering=(0.5, 0.5),
                method=Image.Resampling.LANCZOS
            ).convert(mode='RGB')

    def __migration(self, image_path: str, w: int, h: int) -> Img:
        filename = re.sub('\.[^/.]+$', '', os.path.split(image_path)[1])

        image = ImageOps.fit(
                Image.open(image_path),
                (w, h),
                bleed=0.0,
                centering=(0.5, 0.5),
                method=Image.Resampling.LANCZOS
            ).convert(mode='RGB')

        image.save(
            os.path.join(f'{self.__directory}', f'{filename}.jpg'),
            optimize=True
        )

        try:
            shutil.copy(
                os.path.join(temporary_dataset, f'{filename}.txt'),
                os.path.join(self.__directory, f'{filename}.txt'),
                follow_symlinks=False
            )
        except (FileNotFoundError):
            f = open(
                os.path.join(self.__directory, f'{filename}.txt'),
                'w',
                encoding='UTF-8'
            )
            f.close()

        return image

    def __no_op(self, image_path: str, w: int, h: int) -> Img:
        return Image.open(image_path)

class ImageStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

        self.image_files = []
        [self.image_files.extend(glob.glob(f'{data_dir}' + '/*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]

        self.validator = Validation(
            conf.local.image_store.skip,
            conf.local.image_store.extended
        ).validate

        self.resizer = Resize(conf.local.image_store.resize, conf.local.image_store.no_migration).resize

        self.image_files = [x for x in self.image_files if self.validator(x)]

    def __len__(self) -> int:
        return len(self.image_files)

    # iterator returns images as PIL images and their index in the store
    def __iter__(self) -> Generator[Tuple[Img, int], None, None]:
        for i, f in enumerate(self.image_files):
            yield Image.open(f), i

    # get image by index
    def get_image(self, ref: Tuple[int, int, int]) -> Img:
        return self.resizer(
            self.image_files[ref[0]],
            ref[1],
            ref[2]
        )

    # gets caption by removing the extension from the filename and replacing it with .txt
    def get_caption(self, ref: Tuple[int, int, int]) -> str:
        filename = re.sub('\.[^/.]+$', '', self.image_files[ref[0]]) + '.txt'
        with open(filename, 'r', encoding='UTF-8') as f:
            return f.read()

# for confused questions <contact@lopho.org>
# or via discord <lopho#5445>
class SimpleBucket(torch.utils.data.Sampler):
    """
    Batches samples into buckets of same size.
    """
    def __init__(self,
            store: ImageStore,
            batch_size: int,
            shuffle: bool = True,
            num_replicas: int = 1,
            rank: int = 0,
            resize: bool = False,
            image_side_divisor: int = 64,
            max_image_area: int = 512 ** 2,
            image_side_min: Optional[int] = None,
            image_side_max: Optional[int] = None,
            fixed_size: Optional[tuple[int, int]] = None
    ):
        super().__init__(None)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.store = store
        self.buckets = dict()
        self.ratios = []
        if resize:
            m = image_side_divisor
            assert (max_image_area // m) == max_image_area / m, "resolution not multiple of divisor"
            if image_side_max is not None:
                assert (image_side_max // m) == image_side_max / m, "side not multiple of divisor"
            if image_side_min is not None:
                assert (image_side_min // m) == image_side_min / m, "side not multiple of divisor"
            if fixed_size is not None:
                assert (fixed_size[0] // m) == fixed_size[0] / m, "side not multiple of divisor"
                assert (fixed_size[1] // m) == fixed_size[1] / m, "side not multiple of divisor"
            if image_side_min is None:
                if image_side_max is None:
                    image_side_min = m
                else:
                    image_side_min = max((max_image_area // image_side_max) * m, m)
            if image_side_max is None:
                image_side_max = max((max_image_area // image_side_min) * m, m)
            self.fixed_size = fixed_size
            self.image_side_min = image_side_min
            self.image_side_max = image_side_max
            self.image_side_divisor = image_side_divisor
            self.max_image_area = max_image_area
        self.dropped_samples = []
        self.init_buckets(resize)
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        # generate batches
        batches = []
        for b in self.buckets:
            idxs = self.buckets[b]
            if self.shuffle:
                random.shuffle(idxs)
            rest = len(idxs) % self.batch_size
            idxs = idxs[rest:]
            batched_idxs = [idxs[i:i + self.batch_size] for i in range(0, len(idxs), self.batch_size)]
            for bidx in batched_idxs:
                batches.append([(idx, b[0], b[1]) for idx in bidx])
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches[self.rank::self.num_replicas])

    def __len__(self):
        return self.get_batch_count() // self.num_replicas

    def get_batch_count(self) -> int:
        return reduce(lambda x, y: x + len(y) // self.batch_size, self.buckets.values(), 0)

    def _fit_image_size(self, w, h):
        if self.fixed_size is not None:
            return self.fixed_size
        max_area = self.max_image_area
        scale = (max_area / (w * h)) ** 0.5
        m = self.image_side_divisor
        w2 = round((w * scale) / m) * m
        h2 = round((h * scale) / m) * m
        if w2 * h2 > max_area: # top end can round over limits
            w = int((w * scale) / m) * m
            h = int((h * scale) / m) * m
        else:
            w = w2
            h = h2
        w = min(max(w, self.image_side_min), self.image_side_max)
        h = min(max(h, self.image_side_min), self.image_side_max)
        return w, h

    def init_buckets(self, resize = False):
        # create buckets
        buckets = {}
        for img, idx in tqdm.tqdm(self.store, desc='Bucketing', dynamic_ncols=True):
            key = img.size
            img.close()
            if resize:
                key = self._fit_image_size(*key)
            buckets.setdefault(key, []).append(idx)
        # fit buckets < batch_size in closest bucket if resizing is enabled
        if resize:
            for b in buckets:
                if len(buckets[b]) < self.batch_size:
                    # find closest bucket
                    best_fit = float('inf')
                    best_bucket = None
                    for ob in buckets:
                        if ob == b or len(buckets[ob]) == 0:
                            continue
                        d = abs(ob[0] - b[0]) + abs(ob[1] - b[1])
                        if d < best_fit:
                            best_fit = d
                            best_bucket = ob
                    if best_bucket is not None:
                        buckets[best_bucket].extend(buckets[b])
                        buckets[b].clear()
        # drop buckets < batch_size
        for b in list(buckets.keys()):
            if len(buckets[b]) < self.batch_size:
                self.dropped_samples += buckets.pop(b)
            else:
                self.ratios.append(b[0] / b[1])
        self.buckets = buckets

    def get_bucket_info(self):
        return json.dumps({
                "buckets": list(self.buckets.keys()),
                "ratios": self.ratios
        })

class AspectDataset(torch.utils.data.Dataset):
    def __init__(self, store: ImageStore, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, device: torch.device, ucg: float = 0.1):
        self.store = store
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.ucg = ucg

        #if type(self.text_encoder) is torch.nn.parallel.DistributedDataParallel:
        #    self.text_encoder = self.text_encoder.module

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.store)

    def __getitem__(self, item: Tuple[int, int, int]):
        return_dict = {'pixel_values': None, 'input_ids': None}

        image_file = self.store.get_image(item)

        return_dict['pixel_values'] = self.transforms(image_file)
        if random.random() > self.ucg:
            caption_file = self.store.get_caption(item)
        else:
            caption_file = ''

        return_dict['input_ids'] = caption_file
        return return_dict

    def collate_fn(self, examples):
            pixel_values = torch.stack([example['pixel_values'] for example in examples if example is not None])
            pixel_values.to(memory_format=torch.contiguous_format).float()

            if conf.everyone.extended_chunks < 2:
                max_length = self.tokenizer.model_max_length - 2
                input_ids = [self.tokenizer([example['input_ids']], truncation=True, return_length=True, return_overflowing_tokens=False, padding=False, add_special_tokens=False, max_length=max_length).input_ids for example in examples if example is not None]
            else:
                max_length = self.tokenizer.model_max_length
                max_chunks = conf.everyone.extended_chunks
                input_ids = [self.tokenizer([example['input_ids']], truncation=True, return_length=True, return_overflowing_tokens=False, padding=False, add_special_tokens=False, max_length=(max_length * max_chunks) - (max_chunks * 2)).input_ids[0] for example in examples if example is not None]

            tokens = input_ids

            if conf.everyone.extended_chunks < 2:
                for i, x in enumerate(input_ids):
                    for j, y in enumerate(x):
                        input_ids[i][j] = [self.tokenizer.bos_token_id, *y, *np.full((self.tokenizer.model_max_length - len(y) - 1), self.tokenizer.eos_token_id)]

                if conf.everyone.clip_penultimate:
                    input_ids = [self.text_encoder.text_model.final_layer_norm(self.text_encoder(torch.asarray(input_id).to(self.device), output_hidden_states=True)['hidden_states'][-2])[0] for input_id in input_ids]
                else:
                    input_ids = [self.text_encoder(torch.asarray(input_id).to(self.device), output_hidden_states=True).last_hidden_state[0] for input_id in input_ids]
            else:
                max_standard_tokens = max_length - 2
                max_chunks = conf.everyone.extended_chunks
                max_len = np.ceil(max(len(x) for x in input_ids) / max_standard_tokens).astype(int).item() * max_standard_tokens
                if max_len > max_standard_tokens:
                    z = None
                    for i, x in enumerate(input_ids):
                        if len(x) < max_len:
                            input_ids[i] = [*x, *np.full((max_len - len(x)), self.tokenizer.eos_token_id)]
                    batch_t = torch.tensor(input_ids)
                    chunks = [batch_t[:, i:i + max_standard_tokens] for i in range(0, max_len, max_standard_tokens)]
                    for chunk in chunks:
                        chunk = torch.cat((torch.full((chunk.shape[0], 1), self.tokenizer.bos_token_id), chunk, torch.full((chunk.shape[0], 1), self.tokenizer.eos_token_id)), 1)
                        if z is None:
                            if conf.everyone.clip_penultimate:
                                z = self.text_encoder.text_model.final_layer_norm(self.text_encoder(chunk.to(self.device), output_hidden_states=True)['hidden_states'][-2])
                            else:
                                z = self.text_encoder(chunk.to(self.device), output_hidden_states=True).last_hidden_state
                        else:
                            if conf.everyone.clip_penultimate:
                                z = torch.cat((z, self.text_encoder.text_model.final_layer_norm(self.text_encoder(chunk.to(self.device), output_hidden_states=True)['hidden_states'][-2])), dim=-2)
                            else:
                                z = torch.cat((z, self.text_encoder(chunk.to(self.device), output_hidden_states=True).last_hidden_state), dim=-2)
                    input_ids = z
                else:
                    for i, x in enumerate(input_ids):
                        input_ids[i] = [self.tokenizer.bos_token_id, *x, *np.full((self.tokenizer.model_max_length - len(x) - 1), self.tokenizer.eos_token_id)]
                    if conf.everyone.clip_penultimate:    
                        input_ids = self.text_encoder.text_model.final_layer_norm(self.text_encoder(torch.asarray(input_ids).to(self.device), output_hidden_states=True)['hidden_states'][-2])
                    else:
                        input_ids = self.text_encoder(torch.asarray(input_ids).to(self.device), output_hidden_states=True).last_hidden_state
            input_ids = torch.stack(tuple(input_ids))

            return {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'tokens': tokens
            }

# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    # From CompVis LitEMA implementation
    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

        del self.collected_params
        gc.collect()

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]

def backgroundreport(url, data):
    requests.post(url, json=data)

def setuphivemind():
    if os.path.exists(conf.local.working_path):
        shutil.rmtree(conf.local.working_path)
    os.makedirs(conf.local.working_path)

    if requests.get('http://' + conf.everyone.server + '/info').status_code == 200:
        print("Connection Success")
        serverconfig = json.loads(requests.get('http://' + conf.everyone.server + '/config').content)
        print(serverconfig)
        imgs_per_epoch = int(serverconfig["ImagesPerEpoch"])
        total_epochs = int(serverconfig["Epochs"])
        return(imgs_per_epoch, total_epochs)
    else:
        raise ConnectionError("Unable to connect to server")

def getchunk(server, amount):
    if os.path.isdir(temporary_dataset):
        shutil.rmtree(temporary_dataset)
    os.mkdir(temporary_dataset)
    serverdomain = 'http://' + server
    rtasks_url = serverdomain + '/v1/get/tasks/' + str(amount)
    rtasks = requests.get(rtasks_url).json()
    
    print("Downloading Files")
    pfiles = requests.post(serverdomain + '/v1/get/files', json=rtasks)
    tmpZip = conf.local.working_path + '/tmp.zip'
    open(tmpZip, 'wb').write(pfiles.content)
    
    zipfile.ZipFile(tmpZip, 'r').extractall(temporary_dataset)
    os.remove(tmpZip)
    return(rtasks)

def report(server, tasks):
    preport = requests.post('http://' + server + '/v1/post/epochcount', json=tasks)
    if preport.status_code == 200:
        return True
    else:
        return False

def dataloader(tokenizer, text_encoder, device, world_size, rank):
    # load dataset
    store = ImageStore(temporary_dataset)
    dataset = AspectDataset(store, tokenizer, text_encoder, device, ucg=float(conf.everyone.ucg))
    sampler = SimpleBucket(
            store = store,
            batch_size = conf.local.batch_size,
            shuffle = conf.advanced.buckets.shuffle,
            resize = conf.local.image_store.resize,
            image_side_min = conf.advanced.buckets.side_min,
            image_side_max = conf.advanced.buckets.side_max,
            image_side_divisor = 64,
            max_image_area = conf.everyone.resolution ** 2,
            num_replicas = world_size,
            rank = rank
    )

    print(f'STORE_LEN: {len(store)}')

    # if args.output_bucket_info:
    #     print(sampler.get_bucket_info())

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )
    
    # # Migrate dataset
    # if args.resize and not args.no_migration:
    #     for _, batch in enumerate(train_dataloader):
    #         continue
    #     print(f"Completed resize and migration to '{args.dataset}_cropped' please relaunch the trainer without the --resize argument and train on the migrated dataset.")
    #     exit(0)

    return train_dataloader

def main():
    rank = 0
    # world_size = get_world_size()
    torch.cuda.set_device(rank)

    if rank == 0:
        os.makedirs(conf.local.output_path, exist_ok=True)
        
        mode = 'disabled'
        if conf.local.wandb:
            mode = 'online'
        if conf.local.hf_token is not None:
            os.environ['HF_API_TOKEN'] = conf.local.hf_token
            conf.local.hf_token = None
        run = wandb.init(project=conf.everyone.project_name, name=conf.everyone.project_name, config=vars(args), dir=conf.local.output_path+'/wandb', mode=mode)

        # Inform the user of host, and various versions -- useful for debugging issues.
        print("RUN_NAME:", conf.everyone.project_name)
        print("HOST:", socket.gethostname())
        print("CUDA:", torch.version.cuda)
        print("TORCH:", torch.__version__)
        print("TRANSFORMERS:", transformers.__version__)
        print("DIFFUSERS:", diffusers.__version__)
        print("MODEL:", conf.everyone.model)
        print("FP16:", conf.everyone.fp16)
        print("RESOLUTION:", conf.everyone.resolution)


    if conf.local.hf_token is not None:
        print('It is recommended to set the HF_API_TOKEN environment variable instead of passing it as a command line argument since WandB will automatically log it.')
    else:
        try:
            conf.local.hf_token = os.environ['HF_API_TOKEN']
            print("HF Token set via enviroment variable")
        except Exception:
            print("No HF Token detected in arguments or enviroment variable, setting it to none (as in string)")
            conf.local.hf_token = "none"

    device = torch.device('cuda')

    print("DEVICE:", device)

    # setup fp16 stuff
    scaler = torch.cuda.amp.GradScaler(enabled=conf.everyone.fp16)

    # Set seed
    torch.manual_seed(conf.everyone.seed)
    random.seed(conf.everyone.seed)
    np.random.seed(conf.everyone.seed)
    print('RANDOM SEED:', conf.everyone.seed)
    
    tokenizer = CLIPTokenizer.from_pretrained(conf.everyone.model, subfolder='tokenizer', use_auth_token=conf.local.hf_token)
    text_encoder = CLIPTextModel.from_pretrained(conf.everyone.model, subfolder='text_encoder', use_auth_token=conf.local.hf_token)
    vae = AutoencoderKL.from_pretrained(conf.everyone.model, subfolder='vae', use_auth_token=conf.local.hf_token)
    unet = UNet2DConditionModel.from_pretrained(conf.everyone.model, subfolder='unet', use_auth_token=conf.local.hf_token)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    if not conf.everyone.train_text_encoder:
        text_encoder.requires_grad_(False)

    if conf.local.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if conf.everyone.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if conf.local.xformers:
        unet.set_use_memory_efficient_attention_xformers(True)

    # "The “safer” approach would be to move the model to the device first and create the optimizer afterwards."
    weight_dtype = torch.float16 if conf.everyone.fp16 else torch.float32

    # move models to device
    vae = vae.to(device, dtype=weight_dtype)
    unet = unet.to(device, dtype=torch.float32)
    text_encoder = text_encoder.to(device, dtype=weight_dtype if not conf.everyone.train_text_encoder else torch.float32)

    # unet = torch.nn.parallel.DistributedDataParallel(
    #     unet,
    #     device_ids=[rank],
    #     output_device=rank,
    #     gradient_as_bucket_view=True
    # )

    # if conf.everyone.train_text_encoder:
    #     text_encoder = torch.nn.parallel.DistributedDataParallel(
    #         text_encoder,
    #         device_ids=[rank],
    #         output_device=rank,
    #         gradient_as_bucket_view=True
    #     )

    if conf.local.bit_adam: # Bits and bytes is only supported on certain CUDA setups, so default to regular adam if it fails.
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except:
            print('bitsandbytes not supported, using regular Adam optimizer')
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
    tmp_optimizer = optimizer_cls(
        optimizer_parameters,
        # optimizer_class=optimizer_cls,
        # parameters_as_bucket_view=True,
        lr=float(conf.everyone.lr),
        betas=(float(conf.advanced.opt.betas.one), float(conf.advanced.opt.betas.two)),
        eps=float(conf.advanced.opt.epsilon),
        weight_decay=float(conf.advanced.opt.weight_decay),
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        conf.everyone.model,
        subfolder='scheduler',
        use_auth_token=conf.local.hf_token,
    )

    # Hivemind Setup
    # get network peers (if mother peer then ignore)
    rmaddrs_rq = requests.get('http://' + conf.everyone.server + "/v1/get/peers")
    if rmaddrs_rq.status_code == 200:
        peer_list = json.loads(rmaddrs_rq.content)
    else:
        raise ConnectionError("Unable to obtain peers from server")

    # set local maddrs ports
    host_maddrs_tcp = "/ip4/0.0.0.0/tcp/" + str(conf.local.networking.internal.tcp)
    host_maddrs_udp = "/ip4/0.0.0.0/udp/" + str(conf.local.networking.internal.udp) + "/quic"

    # set public to-be-announced maddrs
    # get public ip
    if conf.local.networking.external.ip == "":
        conf.local.networking.external.ip = None

    if conf.local.networking.external.ip == "auto" or conf.local.networking.external.ip is None:
        completed = False
        if completed is False:
            try:
                ip = requests.get("https://api.ipify.org/", timeout=5).text
                ipsrc = "online"
                completed = True
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
                print("Ipfy.org took too long, trying another domain.")
        if completed is False:
            try:
                ip = requests.get("https://ipv4.icanhazip.com/", timeout=5).text
                ipsrc = "online"
                completed = True
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
                print("Icanhazip.com took too long, trying another domain.")
        if completed is False:
            try:
                tmpjson = json.loads(requests.get("https://jsonip.com/", timeout=5).content)
                ip = tmpjson["ip"]
                ipsrc = "online"
                completed = True
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
                print("Jsonip.com took too long, ran out of alternatives.")
                raise(ConnectionError)
        
    else:
        ip = conf.local.networking.external.ip
        ipsrc = "config"
    
    #check if valid ip
    try:
        ip = ipaddress.ip_address(ip)
        ip = str(ip)
    except Exception:
        raise ValueError("Invalid IP, please check the configuration file. IP Source: " + ipsrc)

    public_maddrs_tcp = "/ip4/" + ip + "/tcp/" + str(conf.local.networking.external.tcp)
    public_maddrs_udp = "/ip4/" + ip + "/udp/" + str(conf.local.networking.external.udp) + "/quic"

    #init dht
    #TODO: add announce_maddrs
    dht = hivemind.DHT(
        host_maddrs=[host_maddrs_tcp, host_maddrs_udp],
        initial_peers=peer_list,
        start=True,
        announce_maddrs=[public_maddrs_tcp, public_maddrs_udp]
    )

    #set compression and optimizer
    compression = Float16Compression()

    lr_scheduler = get_scheduler(
    conf.everyone.lr_scheduler,
    optimizer=tmp_optimizer,
    num_warmup_steps=int(float(conf.advanced.lr_scheduler_warmup) * imgs_per_epoch * total_epochs),
    num_training_steps=total_epochs * imgs_per_epoch,
    )

    optimizer = hivemind.Optimizer(
        dht=dht,
        run_id="testrun",
        batch_size_per_step=1,
        target_batch_size=4000,
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

    print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
    print("Global IP:", hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs()))

    #statistics
    if conf.local.stats.enable:
        statconfig = {"geoaprox": False, "bandwidth": False, "specs": False}
        bandwidthstats = {}
        specs_stats = {}
        print("Stats enabled")

        if conf.local.stats.geoaprox:
            statconfig['geoaprox'] = True

        if conf.local.stats.bandwidth:
            statconfig["bandwidth"] = True
            import speedtest
            session = speedtest.Speedtest()
            download = session.download()
            upload = session.upload()
            ping = session.results.ping
            bandwidthstats = {"download": str(download), "upload": str(upload), "ping": str(ping)}

        if conf.local.stats.specs:
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
        pstats = requests.post('http://' + conf.everyone.server + '/v1/post/stats', json=json.dumps(statsjson))
        if pstats.status_code != 200:
            raise ConnectionError("Failed to report stats")

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
                scheduler=PNDMScheduler.from_pretrained(conf.everyone.model, subfolder="scheduler", use_auth_token=conf.local.hf_token),
                safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
            )
            print(f'saving checkpoint to: {conf.local.output_path}/{"hivemind"}_{global_step}')
            pipeline.save_pretrained(f'{conf.local.output_path}/{"hivemind"}_{global_step}')

            if conf.everyone.use_ema:
                ema_unet.restore(unet.parameters())

    # train!
    try:
        already_done_steps = (optimizer.tracker.global_progress.samples_accumulated + (optimizer.tracker.global_progress.epoch * optimizer.target_batch_size))
        print("Skipping", already_done_steps, "steps on the LR Scheduler.")
        for i in range(already_done_steps):
            lr_scheduler.step()
        print("Done")
        loss = torch.tensor(0.0, device=device, dtype=weight_dtype)
        while True:
            print(get_gpu_ram())
            recipt = getchunk(conf.everyone.server, conf.everyone.imgcount)

            #Note: we removed worldsize here
            train_dataloader = dataloader(tokenizer, text_encoder, device, 1, rank)

            num_steps_per_epoch = len(train_dataloader)
            progress_bar = tqdm.tqdm(range(num_steps_per_epoch), desc="Total Steps", leave=False)
            global_step = 0

            unet.train()
            if conf.everyone.train_text_encoder:
                text_encoder.train()

            for _, batch in enumerate(train_dataloader):
                
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

                if not conf.everyone.train_text_encoder:
                    # Predict the noise residual and compute loss
                    with torch.autocast('cuda', enabled=conf.everyone.fp16):
                        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                    # backprop and update
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                else:
                    # Predict the noise residual and compute loss
                    with torch.autocast('cuda', enabled=conf.everyone.fp16):
                        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                    # backprop and update
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
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
                rank_images_per_second = conf.local.batch_size * steps_per_second
                #world_images_per_second = rank_images_per_second #* world_size
                samples_seen = global_step * conf.local.batch_size #* world_size

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
                    # if counter < 5:
                    #     counter += 1
                    # elif counter >= 5:
                    #     data = {
                    #         "tracker.global_progress": optimizer.tracker.global_progress,
                    #         "tracker.local_progress": optimizer.tracker.local_progress,
                    #     }
                    #     print(data)
                    #     counter = 0
                    #Thread(target=backgroundreport, args=(("http://" + conf.everyone.server + "/v1/post/ping"), "world_images_per_second")).start()

                if global_step % conf.local.save_steps == 0 and global_step > 0:
                    save_checkpoint(global_step)
                    
                if conf.local.inference.enable:
                    if global_step % conf.inference.log_steps == 0 and global_step > 0:
                        if rank == 0:
                            # get prompt from random batch
                            prompt = tokenizer.decode(batch['tokens'][random.randint(0, len(batch['tokens'])-1)])

                            if conf.inference.image_log_scheduler == 'DDIMScheduler':
                                print('using DDIMScheduler scheduler')
                                scheduler = DDIMScheduler.from_pretrained(conf.everyone.model, subfolder="scheduler", use_auth_token=conf.local.hf_token)
                            else:
                                print('using PNDMScheduler scheduler')
                                scheduler=PNDMScheduler.from_pretrained(conf.everyone.model, subfolder="scheduler", use_auth_token=conf.local.hf_token)

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
                            if conf.local.wandb:
                                images = []
                            else:
                                saveInferencePath = conf.local.output_path + "/inference"
                                os.makedirs(saveInferencePath, exist_ok=True)
                            with torch.no_grad():
                                with torch.autocast('cuda', enabled=conf.everyone.fp16):
                                    for _ in range(conf.local.inference.amount):
                                        if conf.local.wandb:
                                            images.append(
                                                wandb.Image(pipeline(
                                                    prompt, num_inference_steps=conf.local.inference.inference_steps
                                                ).images[0],
                                                caption=prompt)
                                            )
                                        else:
                                            from datetime import datetime
                                            images = pipeline(prompt, num_inference_steps=conf.local.inference.inference_steps).images[0]
                                            filenameImg = str(time.time_ns()) + ".png"
                                            filenameTxt = str(time.time_ns()) + ".txt"
                                            images.save(saveInferencePath + "/" + filenameImg)
                                            with open(saveInferencePath + "/" + filenameTxt, 'a') as f:
                                                f.write('Used prompt: ' + prompt + '\n')
                                                f.write('Generated Image Filename: ' + filenameImg + '\n')
                                                f.write('Generated at: ' + str(global_step) + ' steps' + '\n')
                                                f.write('Generated at: ' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+ '\n')

                            # log images under single caption
                            if conf.local.wandb:
                                run.log({'images': images}, step=global_step)

                            # cleanup so we don't run out of memory
                            del pipeline
                            gc.collect()
            sreport = report(conf.everyone.server, recipt)
            if sreport is True:
                print("Report Success")
            else:
                raise ConnectionError("Couldn't report")
    except Exception as e:
        print(f'Exception caught on rank {rank} at step {global_step}, saving checkpoint...\n{e}\n{traceback.format_exc()}')
        pass

    save_checkpoint(global_step)

    #cleanup()

    print(get_gpu_ram())
    print('Done!')

if __name__ == "__main__":
    #setup()
    imgs_per_epoch, total_epochs = setuphivemind()
    main()
