import pytorch_lightning as pl
import torch
import argparse
import time
import logging

from lib.model import load_model

from hivemind import Float16Compression, Uniform8BitQuantization
from hivemind.compression import SizeAdaptiveCompression
from omegaconf import OmegaConf

from pytorch_lightning.loggers import wandb
from pytorch_lightning.strategies import HivemindStrategy

def gt():
    return(time.time_ns())

logging.basicConfig(filename=("log-" + str(gt()) + ".txt") , encoding='utf-8', level=logging.DEBUG)

#logging functions
ld = logging.debug
li = logging.info
lw = logging.warning
le = logging.error

parser = argparse.ArgumentParser(description="Waifu Diffusion Finetuner ported to Lightning")
parser.add_argument('-c', '--config', type=str, required=True, help="Path to the configuration file")
parser.add_argument('-p', '--peers', type=str, required=False, default=None, help="Hivemind peers")
args = parser.parse_args()

print("PEERS: " + args.peers)

pathToConf = args.config
li('Loading configuration file from ' + str(pathToConf))
config = OmegaConf.load(pathToConf)
li('Load Sucess')

ld('Defining functions')

def main():
    torch.manual_seed(config.trainer.seed)

    li('Loading model')
    model = load_model(config)

    ld('Initiating logger')
    if config.logger.enable:
        logger = (
            wandb.WandbLogger(project=config.logger.wandb_id, name=str(gt()))
        )
    else:
        logger = None
    
    if config.hivemind:
        li('Initiating Hivemind Configuration')
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
                initial_peers=[args.peers]
            )
        )
    else:
        strategy = None

    li('Initiating Trainer')
    trainer = pl.Trainer(
    logger = logger,
    strategy = strategy,
    reload_dataloaders_every_n_epochs=1,
    **config.lightning
    )

    trainer.tune(model=model)
    li('Initiating Trainer Fit')
    trainer.fit(
        model=model,
        ckpt_path=None,
    )

if __name__ == "__main__":
    main()



