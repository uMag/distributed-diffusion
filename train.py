import pytorch_lightning as pl
import torch
import argparse
import time

from transformers import CLIPTokenizer
from data.engines import ImageStore, AspectBucket, AspectBucketSampler, AspectDataset
from lib.model import load_model
from hivemind import Float16Compression, Uniform8BitQuantization
from hivemind.compression import SizeAdaptiveCompression
from omegaconf import OmegaConf

from pytorch_lightning.loggers import wandb
from pytorch_lightning.strategies import HivemindStrategy

parser = argparse.ArgumentParser(description="Waifu Diffusion Finetuner ported to Lightning")
parser.add_argument('-c', '--config', type=str, required=True, help="Path to the configuration file")
parser.add_argument('-p', '--peers', type=str, required=False, default=None, help="Hivemind peers")
args = parser.parse_args()

pathToConf = args.config
print(pathToConf)
config = OmegaConf.load(pathToConf)

def gt():
    return(time.time_ns())

def main():
    torch.manual_seed(config.trainer.seed)
    pathToModelDiffuser = config.checkpoint.input.diffusers_path
    resolution = config.dataset.resolution

    tokenizer = CLIPTokenizer.from_pretrained(pathToModelDiffuser, subfolder="tokenizer")

    #do as haru's rather than naifus
    #load dataset
    store = ImageStore(config.dataset.path)
    dataset = AspectDataset(store, tokenizer)
    bucket = AspectBucket(
        store=store,
        num_buckets=config.dataset.buckets.num_buckets,
        batch_size=config.trainer.batch_size,
        bucket_side_min=config.dataset.buckets.bucket_side.min,
        bucket_side_max=config.dataset.buckets.bucket_side.max,
        bucket_side_increment=64,
        max_image_area=int(resolution * resolution),
        max_ratio=2.0
    )
    sampler = AspectBucketSampler(
        bucket=bucket,
        num_replicas=1, #because we are not doing distributed and thats the default
        rank=0, #same reason as above
    )

    print(f'STORE_LEN: {len(store)}')

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )

    model = load_model(config, len(train_dataloader), tokenizer)

    if config.logger.enable:
        logger = (
            wandb.WandbLogger(project=config.logger.wandb_id, name=str(gt()))
        )
    else:
        logger = None

    if config.hivemind:
        list1 = model.configure_optimizers()

        compression = SizeAdaptiveCompression(
            threshold=2 ** 16 + 1, less=Float16Compression(), greater_equal=Uniform8BitQuantization()
        )

        strategy = (
            HivemindStrategy(
                scheduler_fn=list1[1],
                grad_compression=compression,
                state_averaging_compression=compression,
                target_batch_size=1000,
                verbose=True,
                run_id="testrun",
                host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
                reuse_grad_buffers=True,
                offload_optimizer=False,
                initial_peers=args.peers
            )
        )
    else:
        strategy = None

    trainer = pl.Trainer(
    logger = logger,
    strategy = strategy,
    **config.lightning
    )

    trainer.tune(model=model)
    trainer.fit(
        model=model,
        ckpt_path=None,
        train_dataloaders=train_dataloader
    )

if __name__ == "__main__":
    main()



