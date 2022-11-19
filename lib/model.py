import pytorch_lightning as pl
import torch
import time
import logging

from transformers import CLIPTextModel, CLIPFeatureExtractor, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline, PNDMScheduler
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from data.engines import ImageStore, AspectBucket, AspectBucketSampler, AspectDataset

from .depos.handler import DeposHandler

#logging functions
ld = logging.debug
li = logging.info
lw = logging.warning
le = logging.error

def tp(input: str):
    if input == "fp16":
        precision = torch.float16
    elif input == "fp32":
        precision = torch.float32
    return precision

class StableDiffusionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        ld('Init function of StableDiffusionModel class')
        self.config = config
        self.input_model_path = self.config.checkpoint.input.diffusers_path
        use_auth = self.config.use_auth_token

        ld('Calling server...')
        self.DeposObject = DeposHandler(self.config)

        ld('input_model_path: ' + str(self.input_model_path))
        self.output_model_path = self.config.checkpoint.output.output_path
        ld('output_model_path: ' + str(self.output_model_path))

        li('Loading tokenizer')
        self.tokenizer = CLIPTokenizer.from_pretrained(self.input_model_path, subfolder="tokenizer", use_auth_token=use_auth)

        self.text_encoder = CLIPTextModel.from_pretrained(self.input_model_path, subfolder='text_encoder', use_auth_token=use_auth) #L721
        ld('Text Encoder set')
        self.vae = AutoencoderKL.from_pretrained(self.input_model_path, subfolder='vae', use_auth_token=use_auth)
        ld('VAE set')
        self.unet = UNet2DConditionModel.from_pretrained(self.input_model_path, subfolder='unet', use_auth_token=use_auth)
        ld('Unet set')

        #Unlike naifu's trainer, diffusers trainer starts a new DDPMScheduler rather than loading it from the model.
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule='scaled_linear',
            num_train_timesteps=1000,
            clip_sample=False
        )
        ld('Noise Scheduler set')

        unet_prec = config.trainer.unet.precision
        vae_prec = config.trainer.vae.precision
        te_prec = config.trainer.text_encoder.precision
        ld('Precision Conversion Stage')

        self.unet.to(tp(unet_prec))
        ld('Unet precision set to ' + str(unet_prec))

        self.vae.to(tp(vae_prec))
        ld('VAE precision set to ' + str(vae_prec))

        self.text_encoder.to(tp(te_prec))
        ld('Text Encoder precision set to ' + str(te_prec))

        #Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        ld('Freezed VAE and Text Encoder')

        if config.trainer.gradient_checkpointing:
            ld('Enabling Gradient Checkpointing')
            self.unet.enable_gradient_checkpointing()
            ld('Gradient Checkpointing Success')

        if config.trainer.use_xformers:
            ld('Enabling XFormers Use')
            self.unet.set_use_memory_efficient_attention_xformers(True)
            ld('XFormers Success')

    def training_step(self, batch, batch_idx):
        li('training_step, batch_idx is ' + str(batch_idx))
        b_start = time.perf_counter()
        # Convert images to latent space
        #b_start = time.perf_counter()
        ld('Latents Stage Init')
        latents = self.vae.encode(batch['pixel_values'].to(dtype=tp(self.config.trainer.latents.precision))).latent_dist.sample()
        latents = latents * 0.18215 # le magic number

        #Naifu's trainer puts the text embedding here but to follow the 
        #diffusers trainer's we are not going to do that

        # Sample noise
        ld('Sample Noise')
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample a random timestep for each image         Ignore the config part (naifu)
        ld('Random Timestep Init')
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        ld('Add noise to the latents')
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps).to(tp(self.config.trainer.n_latents.precision))

        #Following diffusers trainer:
        # Get the text embedding for conditioning
        ld('Obtaining Text Embedding')
        encoder_hidden_states = self.text_encoder(batch['input_ids'], output_hidden_states=True)
        #  TODO: Add entry to config file
        enable_clip_penultimate = False
        if enable_clip_penultimate:
            ld('Clip Penultimate Init')
            encoder_hidden_states = self.text_encoder.text_model.final_layer_norm(encoder_hidden_states['hidden_states'][-2])
        else:
            ld('No Clip Penultimate Init')
            encoder_hidden_states = encoder_hidden_states.last_hidden_state

        # Predict noise residual
        # TODO: haru's uses torch.autocast, naifu doesn't.
        ld('Predicting Noise Residual Init')
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states.to(tp(self.config.trainer.n_pred.precision))).sample

        # Compute loss
        # TODO: haru's uses torch.nn.function, naifu does too but imports it as "F"
        ld('Compute Loss Init')
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        #perf
        ld('Perf Calc Init')
        b_end = time.perf_counter()
        seconds_per_step = b_end - b_start
        steps_per_second = 1 / seconds_per_step
        images_per_second = self.config.trainer.batch_size * steps_per_second
        ld('Perf Calc End')

        #naifu trainer only
        self.log("train_loss", loss)
        self.log("train_steps_per_second", steps_per_second)
        self.log("images_per_second", images_per_second)
        self.log("seconds_per_step", seconds_per_step)
        return loss

    def configure_optimizers(self):
        ld('configure_optimizers')
        oc = self.config.trainer.optimizer #oc: opt conf
        ld(oc)
        lsc = self.config.trainer.lr_scheduler #lsc: lr scheduler conf
        ld(lsc)
        
        if self.config.lightning.auto_lr_find:
            li("Auto Find lr Enabled")
            oc.lr = self.lr
        
        #almost the same as L589, L579-587 sets the optimizers,
        if self.config.trainer.optimizer.use_8bit_adam:
            li("Use 8bit adam enabled")
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
                li('Import Success')
            except:
                print('bitsandbytes not supported, using regular Adam optimizer')
                optimizer_cls = torch.optim.AdamW
                li('Import Fail... using standard Adam')
        else:
            optimizer_cls = torch.optim.AdamW
            li('Using standard Adam')
        # L589 just initiates with the settings        

        optimizer = optimizer_cls(
            self.unet.parameters(),             #on haru's this is
            lr=oc.lr,  #args.lr
            betas=(oc.adam_beta1, oc.adam_beta2),                  #args.adam_beta1 (def:0.9), args.adam_beta2 (def:0.999)
            eps=float(oc.adam_epsilon),                    #args.adam_epsilon (def: 1e-08 as float)
            weight_decay=float(oc.adam_weight_decay),            #args.adam_weight_decay (def: 1e-2 as float)
        )
        ld('Optimizer Set')

        # Assuming scheduler (on naifus) is the same as lr_scheduler (as harus)
        # Important Note: To not to be confused with noise scheduler! check L646

        #Also known as lr_scheduler
        self.train_dataloader_len = self.config.depos.localcapacity
        ld('train_dataloader_len: ' + str(self.train_dataloader_len))

        scheduler = get_scheduler(
            lsc.type,
            optimizer=optimizer,
            num_warmup_steps=float(lsc.warmup) * int(self.train_dataloader_len) * int(self.config.trainer.epochs), #This part is going to be hardcoded until we get a way to get dataset length
            num_training_steps=int(self.config.trainer.epochs) * int(self.train_dataloader_len)
        )
        ld('Scheduler Set')

        return [[optimizer], [scheduler]]
    
    def on_save_checkpoint(self, checkpoint):
        ld('Saving Checkpoint')
        pipeline = StableDiffusionPipeline(
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
            tokenizer=self.tokenizer,
            scheduler=PNDMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
            ),
            safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        )
        print("Saving checkpoint to:" + self.output_model_path)
        li("Saving checkpoint to:" + self.output_model_path)
        pipeline.save_pretrained(self.output_model_path)
        li('Saved')

    def train_dataloader(self):
        li('Loading Dataset')

        li('Downloading Dataset')
        self.recipt, self.datasetPath = self.DeposObject.gather()

        ld('Initiating ImageStore')
        store = ImageStore(self.datasetPath)
        ld('Initiating AspectDataset')
        dataset = AspectDataset(store, self.tokenizer)
        ld('Initiating AspectBucket')

        resolution = self.config.dataset.resolution
        bucket = AspectBucket(
            store=store,
            num_buckets=self.config.dataset.buckets.num_buckets,
            batch_size=self.config.trainer.batch_size,
            bucket_side_min=self.config.dataset.buckets.bucket_side.min,
            bucket_side_max=self.config.dataset.buckets.bucket_side.max,
            bucket_side_increment=64,
            max_image_area=int(resolution * resolution),
            max_ratio=2.0
        )
        ld('Initiating AspectBucketSampler')
        sampler = AspectBucketSampler(
            bucket=bucket,
            num_replicas=1, #because we are not doing distributed and thats the default
            rank=0, #same reason as above
        )
        li('Dataset Loaded')

        li(f'STORE_LEN: {len(store)}')

        ld('Passing Dataloader')
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=dataset.collate_fn
        )

        return(train_dataloader)

    def training_epoch_end(self, outputs):
        li('Finished Epoch!')
        self.DeposObject.report(self.recipt)

            
    #skipping on_train_start, on_train_batch_end, because they use ema things that I dont use
    #skipping on_load_checkpoint bcuz according to docs we might not use them

def get_class(name: str):
    import importlib

    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)

def load_model(config):
    #the naifu guy downloads the model from a url but im not going to add that
    return StableDiffusionModel(config)