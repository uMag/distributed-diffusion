import torch
import torchvision
import os
import glob
import random
import tqdm
import pynvml
import gc
import numpy as np
import json
import re
import shutil
from typing import Optional
from functools import reduce
try:
    pynvml.nvmlInit()
except pynvml.nvml.NVMLError_LibraryNotFound:
    pynvml = None

from typing import Iterable
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageOps
from PIL.Image import Image as Img

from typing import Generator, Tuple

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
        except Exception as error:
            print(f'WARNING: Image cannot be opened: {error}')
            return False

    def __no_op(self, fp: str) -> bool:
        return True

class Resize():
    def __init__(self, is_resizing: bool, is_not_migrating: bool, datasetpath) -> None:
        if not is_resizing:
            self.resize = self.__no_op
            return

        if not is_not_migrating:
            self.resize = self.__migration
            dataset_path = os.path.split(datasetpath)
            self.datasetpath = datasetpath
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
                os.path.join(self.datasetpath, f'{filename}.txt'),
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
    def __init__(self, data_dir: str, conf) -> None:
        self.data_dir = data_dir

        self.image_files = []
        [self.image_files.extend(glob.glob(f'{data_dir}' + '/*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]

        self.validator = Validation(
            conf.image_store_skip,
            conf.image_store_extended
        ).validate

        self.resizer = Resize(conf.image_store_resize, conf.image_store_no_migration, data_dir).resize

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
            fixed_size: Optional[tuple(int, int)] = None
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
    def __init__(self, store: ImageStore, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, device: torch.device, conf, ucg: float = 0.1):
        self.store = store
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.conf = conf
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

            if self.conf.everyone.extended_chunks < 2:
                max_length = self.tokenizer.model_max_length - 2
                input_ids = [self.tokenizer([example['input_ids']], truncation=True, return_length=True, return_overflowing_tokens=False, padding=False, add_special_tokens=False, max_length=max_length).input_ids for example in examples if example is not None]
            else:
                max_length = self.tokenizer.model_max_length
                max_chunks = self.conf.everyone.extended_chunks
                input_ids = [self.tokenizer([example['input_ids']], truncation=True, return_length=True, return_overflowing_tokens=False, padding=False, add_special_tokens=False, max_length=(max_length * max_chunks) - (max_chunks * 2)).input_ids[0] for example in examples if example is not None]

            tokens = input_ids

            if self.conf.everyone.extended_chunks < 2:
                for i, x in enumerate(input_ids):
                    for j, y in enumerate(x):
                        input_ids[i][j] = [self.tokenizer.bos_token_id, *y, *np.full((self.tokenizer.model_max_length - len(y) - 1), self.tokenizer.eos_token_id)]

                if self.conf.everyone.clip_penultimate:
                    input_ids = [self.text_encoder.text_model.final_layer_norm(self.text_encoder(torch.asarray(input_id).to(self.device), output_hidden_states=True)['hidden_states'][-2])[0] for input_id in input_ids]
                else:
                    input_ids = [self.text_encoder(torch.asarray(input_id).to(self.device), output_hidden_states=True).last_hidden_state[0] for input_id in input_ids]
            else:
                max_standard_tokens = max_length - 2
                max_chunks = self.conf.everyone.extended_chunks
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
                            if self.conf.everyone.clip_penultimate:
                                z = self.text_encoder.text_model.final_layer_norm(self.text_encoder(chunk.to(self.device), output_hidden_states=True)['hidden_states'][-2])
                            else:
                                z = self.text_encoder(chunk.to(self.device), output_hidden_states=True).last_hidden_state
                        else:
                            if self.conf.everyone.clip_penultimate:
                                z = torch.cat((z, self.text_encoder.text_model.final_layer_norm(self.text_encoder(chunk.to(self.device), output_hidden_states=True)['hidden_states'][-2])), dim=-2)
                            else:
                                z = torch.cat((z, self.text_encoder(chunk.to(self.device), output_hidden_states=True).last_hidden_state), dim=-2)
                    input_ids = z
                else:
                    for i, x in enumerate(input_ids):
                        input_ids[i] = [self.tokenizer.bos_token_id, *x, *np.full((self.tokenizer.model_max_length - len(x) - 1), self.tokenizer.eos_token_id)]
                    if self.conf.everyone.clip_penultimate:    
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
