#Preprocess dataset and save to parquet files for upload to HF
import argparse
import torch
import torchvision
import glob
import random
import tqdm
import pynvml
import itertools
import numpy as np

try:
    pynvml.nvmlInit()
except pynvml.nvml.NVMLError_LibraryNotFound:
    pynvml = None

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from typing import Dict, List, Generator, Tuple
from scipy.interpolate import interp1d
parser = argparse.ArgumentParser(description="Pre-Process dataset")
parser.add_argument('--model', type=str, default=None, required=True, help='The name of the model to use for finetuning. Could be HuggingFace ID or a directory')
parser.add_argument('--run_name', type=str, default=None, required=True, help='Name of the finetune run.')
parser.add_argument('--dataset', type=str, default=None, required=True, help='The path to the dataset to use for finetuning.')
parser.add_argument('--hivemind', dest='hivemind', default=None, help='Enable hivemind distributed training (ALPHA).')
parser.add_argument('--hfstream', dest='hfstream', default=None, help='Stream dataset from HuggingFace.')
parser.add_argument('--bucket_side_min', type=int, default=256, help='The minimum side length of a bucket.')
parser.add_argument('--bucket_side_max', type=int, default=768, help='The maximum side length of a bucket.')
parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--use_ema', type=bool, default=False, help='Use EMA for finetuning')
parser.add_argument('--ucg', type=float, default=0.1, help='Percentage chance of dropping out the text condition per batch. Ranges from 0.0 to 1.0 where 1.0 means 100% text condition dropout.') # 10% dropout probability
parser.add_argument('--gradient_checkpointing', dest='gradient_checkpointing', type=bool, default=False, help='Enable gradient checkpointing')
parser.add_argument('--use_8bit_adam', dest='use_8bit_adam', type=bool, default=False, help='Use 8-bit Adam optimizer')
parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam beta1')
parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam beta2')
parser.add_argument('--adam_weight_decay', type=float, default=1e-2, help='Adam weight decay')
parser.add_argument('--adam_epsilon', type=float, default=1e-08, help='Adam epsilon')
parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator, this is to be used for reproduceability purposes.')
parser.add_argument('--output_path', type=str, default='./output', help='Root path for all outputs.')
parser.add_argument('--save_steps', type=int, default=500, help='Number of steps to save checkpoints at.')
parser.add_argument('--resolution', type=int, default=512, help='Image resolution to train against. Lower res images will be scaled up to this resolution and higher res images will be scaled down.')
#parser.add_argument('--shuffle', dest='shuffle', type=bool, default=True, help='Shuffle dataset') #Not in use for some reason, need confirmation
parser.add_argument('--hf_token', type=str, default=None, required=False, help='A HuggingFace token is needed to download private models for training.')
parser.add_argument('--project_id', type=str, default='diffusers', help='Project ID for reporting to WandB')
parser.add_argument('--fp16', dest='fp16', type=bool, default=False, help='Train in mixed precision')
parser.add_argument('--image_log_steps', type=int, default=100, help='Number of steps to log images at.')
parser.add_argument('--image_log_amount', type=int, default=4, help='Number of images to log every image_log_steps')
parser.add_argument('--peers', type=str, default=None, nargs="*", help='MUST BE PASSED AS A LIST! ex.: --peers /ipv4/1.1.1.1 /ipv4/2.2.2.2 | Multiaddrs of one or more active DHT peers. If none it will start a new session.')
args = parser.parse_args()

# store = ImageStore(args.dataset)

def _sort_by_ratio(bucket: tuple) -> float:
    return bucket[0] / bucket[1]

def _sort_by_area(bucket: tuple) -> float:
    return bucket[0] * bucket[1]

class ImageStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

        self.image_files = []
        [self.image_files.extend(glob.glob(f'{data_dir}' + '/*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]

    def __len__(self) -> int:
        return len(self.image_files)

    # iterator returns images as PIL images and their index in the store
    def entries_iterator(self) -> Generator[Tuple[Image.Image, int], None, None]:
        for f in range(len(self)):
            yield Image.open(self.image_files[f]), f

    # get image by index
    def get_image(self, index: int) -> Image.Image:
        return Image.open(self.image_files[index])
    
    # gets caption by removing the extension from the filename and replacing it with .txt
    def get_caption(self, index: int) -> str:
        filename = self.image_files[index].split('.')[0] + '.txt'
        with open(filename, 'r') as f:
            return f.read()

store = ImageStore(args.dataset)
tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder='tokenizer', use_auth_token=args.hf_token)

class AspectDataset(torch.utils.data.Dataset):
    def __init__(self, store: ImageStore, tokenizer: CLIPTokenizer, ucg: float = 0.1):
        self.store = store
        self.tokenizer = tokenizer
        self.ucg = ucg
        
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.store)

    def __getitem__(self, item: int):
        return_dict = {'pixel_values': None, 'input_ids': None}

        image_file = self.store.get_image(item)
        return_dict['pixel_values'] = self.transforms(image_file)
        if random.random() > self.ucg:
            caption_file = self.store.get_caption(item)
        else:
            caption_file = ''
        return_dict['input_ids'] = self.tokenizer(caption_file, max_length=self.tokenizer.model_max_length, padding='do_not_pad', truncation=True).input_ids

        return return_dict

    
    def collate_fn(self, examples):
            pixel_values = torch.stack([example['pixel_values'] for example in examples if example is not None])
            pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids = [example['input_ids'] for example in examples if example is not None]
            padded_tokens = self.tokenizer.pad({'input_ids': input_ids}, return_tensors='pt', padding=True)
            return {
                'pixel_values': pixel_values,
                'input_ids': padded_tokens.input_ids,
                'attention_mask': padded_tokens.attention_mask,
            }

dataset = AspectDataset(store, tokenizer)

class AspectBucket:
    def __init__(self, store: ImageStore,
                 num_buckets: int,
                 batch_size: int,
                 bucket_side_min: int = 256,
                 bucket_side_max: int = 768,
                 bucket_side_increment: int = 64,
                 max_image_area: int = 512 * 768,
                 max_ratio: float = 2):
        
        self.requested_bucket_count = num_buckets
        self.bucket_length_min = bucket_side_min
        self.bucket_length_max = bucket_side_max
        self.bucket_increment = bucket_side_increment
        self.max_image_area = max_image_area
        self.batch_size = batch_size
        self.total_dropped = 0

        if max_ratio <= 0:
            self.max_ratio = float('inf')
        else:
            self.max_ratio = max_ratio
        
        self.store = store
        self.buckets = []
        self._bucket_ratios = []
        self._bucket_interp = None
        self.bucket_data: Dict[tuple, List[int]] = dict()
        self.init_buckets()
        self.fill_buckets()
    
    def init_buckets(self):
        possible_lengths = list(range(self.bucket_length_min, self.bucket_length_max + 1, self.bucket_increment))
        possible_buckets = list((w, h) for w, h in itertools.product(possible_lengths, possible_lengths)
                        if w >= h and w * h <= self.max_image_area and w / h <= self.max_ratio)
        
        buckets_by_ratio = {}

        # group the buckets by their aspect ratios
        for bucket in possible_buckets:
            w, h = bucket
            # use precision to avoid spooky floats messing up your day
            ratio = '{:.4e}'.format(w / h)

            if ratio not in buckets_by_ratio:
                group = set()
                buckets_by_ratio[ratio] = group
            else:
                group = buckets_by_ratio[ratio]

            group.add(bucket)

        # now we take the list of buckets we generated and pick the largest by area for each (the first sorted)
        # then we put all of those in a list, sorted by the aspect ratio
        # the square bucket (LxL) will be the first
        unique_ratio_buckets = sorted([sorted(buckets, key=_sort_by_area)[-1]
                                       for buckets in buckets_by_ratio.values()], key=_sort_by_ratio)

        # how many buckets to create for each side of the distribution
        bucket_count_each = int(np.clip((self.requested_bucket_count + 1) / 2, 1, len(unique_ratio_buckets)))

        # we know that the requested_bucket_count must be an odd number, so the indices we calculate
        # will include the square bucket and some linearly spaced buckets along the distribution
        indices = {*np.linspace(0, len(unique_ratio_buckets) - 1, bucket_count_each, dtype=int)}

        # make the buckets, make sure they are unique (to remove the duplicated square bucket), and sort them by ratio
        # here we add the portrait buckets by reversing the dimensions of the landscape buckets we generated above
        buckets = sorted({*(unique_ratio_buckets[i] for i in indices),
                          *(tuple(reversed(unique_ratio_buckets[i])) for i in indices)}, key=_sort_by_ratio)

        self.buckets = buckets

        # cache the bucket ratios and the interpolator that will be used for calculating the best bucket later
        # the interpolator makes a 1d piecewise interpolation where the input (x-axis) is the bucket ratio,
        # and the output is the bucket index in the self.buckets array
        # to find the best fit we can just round that number to get the index
        self._bucket_ratios = [w / h for w, h in buckets]
        self._bucket_interp = interp1d(self._bucket_ratios, list(range(len(buckets))), assume_sorted=True,
                                       fill_value=None)

        for b in buckets:
            self.bucket_data[b] = []
        
    def get_batch_count(self):
        return sum(len(b) // self.batch_size for b in self.bucket_data.values())
    
    def get_batch_iterator(self) -> Generator[Tuple[Tuple[int, int], List[int]], None, None]:
        """
        Generator that provides batches where the images in a batch fall on the same bucket

        Each element generated will be:
            ((w, h), [image1, image2, ..., image{batch_size}])

        where each image is an index into the dataset
        :return:
        """
        max_bucket_len = max(len(b) for b in self.bucket_data.values())
        index_schedule = list(range(max_bucket_len))
        random.shuffle(index_schedule)

        bucket_len_table = {
            b: len(self.bucket_data[b]) for b in self.buckets
        }

        bucket_schedule = []
        for i, b in enumerate(self.buckets):
            bucket_schedule.extend([i] * (bucket_len_table[b] // self.batch_size))

        random.shuffle(bucket_schedule)

        bucket_pos = {
            b: 0 for b in self.buckets
        }

        total_generated_by_bucket = {
            b: 0 for b in self.buckets
            }

        for bucket_index in bucket_schedule:
            b = self.buckets[bucket_index]
            i = bucket_pos[b]
            bucket_len = bucket_len_table[b]

            batch = []
            while len(batch) != self.batch_size:
                # advance in the schedule until we find an index that is contained in the bucket
                k = index_schedule[i]
                if k < bucket_len:
                    entry = self.bucket_data[b][k]
                    batch.append(entry)

                i += 1

            total_generated_by_bucket[b] += self.batch_size
            bucket_pos[b] = i
            yield [idx for idx in batch]
    
    def fill_buckets(self):
        entries = self.store.entries_iterator()
        total_dropped = 0

        for entry, index in tqdm.tqdm(entries, total=len(self.store)):
            if not self._process_entry(entry, index):
                total_dropped += 1

        for b, values in self.bucket_data.items():
            # shuffle the entries for extra randomness and to make sure dropped elements are also random
            random.shuffle(values)

            # make sure the buckets have an exact number of elements for the batch
            to_drop = len(values) % self.batch_size
            self.bucket_data[b] = list(values[:len(values) - to_drop])
            total_dropped += to_drop

        self.total_dropped = total_dropped
    
    def _process_entry(self, entry: Image.Image, index: int) -> bool:
        aspect = entry.width / entry.height
        
        if aspect > self.max_ratio or (1 / aspect) > self.max_ratio:
            return False
        
        best_bucket = self._bucket_interp(aspect)

        if best_bucket is None:
            return False
        
        bucket = self.buckets[round(float(best_bucket))]

        self.bucket_data[bucket].append(index)

        return True


bucket = AspectBucket(store, 16, args.batch_size, args.bucket_side_min, args.bucket_side_max, 64, args.resolution * args.resolution, 2.0)

class AspectBucketSampler(torch.utils.data.Sampler):
    def __init__(self, bucket: AspectBucket):
        super().__init__(None)
        self.bucket = bucket
    
    def __iter__(self):
        yield from self.bucket.get_batch_iterator()
    
    def __len__(self):
        return self.bucket.get_batch_count()

sampler = AspectBucketSampler(bucket)

print(f'STORE_LEN: {len(store)}')

train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=4,
    collate_fn=dataset.collate_fn
)

print(type(train_dataloader))

weight_dtype = torch.float32

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder='text_encoder', use_auth_token=args.hf_token)
vae = AutoencoderKL.from_pretrained(args.model, subfolder='vae', use_auth_token=args.hf_token)
unet = UNet2DConditionModel.from_pretrained(args.model, subfolder='unet', use_auth_token=args.hf_token)

# move models to device
vae = vae.to(device, dtype=weight_dtype)
unet = unet.to(device, dtype=torch.float32)
text_encoder = text_encoder.to(device, dtype=weight_dtype)

count = 0

for step, batch in enumerate(train_dataloader):
    count = count + 1
    print("Restart...")
    latents = None
    # Convert images to latent space
    print("LATENT")
    latents = batch['pixel_values']
    print("TEXT")
    text = batch["input_ids"]

    print("counter: " + str(count))
# dataset = AspectDataset(store, tokenizer)