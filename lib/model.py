import pytorch_lightning as pl
import torch
import time

from transformers import CLIPTextModel, CLIPFeatureExtractor
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline, PNDMScheduler
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

def tp(input: str):
    if input == "fp16":
        precision = torch.float16
    elif input == "fp32":
        precision = torch.float32
    return precision

class StableDiffusionModel(pl.LightningModule):
    def __init__(self, config, train_dataloader_len, tokenizer):
        super().__init__()
        self.config = config
        self.train_dataloader_len = train_dataloader_len
        self.input_model_path = config.checkpoint.input.diffusers_path
        self.output_model_path = config.checkpoint.output.output_path

        self.tokenizer = tokenizer

        self.text_encoder = CLIPTextModel.from_pretrained(self.input_model_path, subfolder='text_encoder') #L721
        self.vae = AutoencoderKL.from_pretrained(self.input_model_path, subfolder='vae')
        self.unet = UNet2DConditionModel.from_pretrained(self.input_model_path, subfolder='unet')

        #Unlike naifu's trainer, diffusers trainer starts a new DDPMScheduler rather than loading it from the model.
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule='scaled_linear',
            num_train_timesteps=1000,
            clip_sample=False
        )

        self.unet.to(tp(config.trainer.unet.precision))

        self.vae.to(tp(config.trainer.vae.precision))
        self.text_encoder.to(tp(config.trainer.text_encoder.precision))

        #Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        if config.trainer.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        if config.trainer.use_xformers:
            self.unet.set_use_memory_efficient_attention_xformers(True)

    def training_step(self, batch, batch_idx):
        b_start = time.perf_counter()
        # Convert images to latent space
        #b_start = time.perf_counter()
        latents = self.vae.encode(batch['pixel_values'].to(dtype=tp(self.config.trainer.latents.precision))).latent_dist.sample()
        latents = latents * 0.18215 # le magic number

        #Naifu's trainer puts the text embedding here but to follow the 
        #diffusers trainer's we are not going to do that

        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample a random timestep for each image         Ignore the config part (naifu)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps).to(tp(self.config.trainer.n_latents.precision))

        #Following diffusers trainer:
        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(batch['input_ids'], output_hidden_states=True)
        #  TODO: Add entry to config file
        enable_clip_penultimate = False
        if enable_clip_penultimate:
            encoder_hidden_states = self.text_encoder.text_model.final_layer_norm(encoder_hidden_states['hidden_states'][-2])
        else:
            encoder_hidden_states = encoder_hidden_states.last_hidden_state

        # Predict noise residual
        # TODO: haru's uses torch.autocast, naifu doesn't.
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states.to(tp(self.config.trainer.n_pred.precision))).sample

        # Compute loss
        # TODO: haru's uses torch.nn.function, naifu does too but imports it as "F"
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        #perf
        b_end = time.perf_counter()
        seconds_per_step = b_end - b_start
        steps_per_second = 1 / seconds_per_step
        images_per_second = self.config.trainer.batch_size * steps_per_second
        
        #naifu trainer only
        self.log("train_loss", loss)
        self.log("train_steps_per_second", steps_per_second)
        self.log("images_per_second", images_per_second)
        self.log("seconds_per_step", seconds_per_step)
        return loss

    def configure_optimizers(self):
        oc = self.config.trainer.optimizer #oc: opt conf
        lsc = self.config.trainer.lr_scheduler #lsc: lr scheduler conf

        if self.config.lightning.auto_lr_find:
            oc.lr = self.lr
        
        #almost the same as L589, L579-587 sets the optimizers,
        if self.config.trainer.optimizer.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
            except:
                print('bitsandbytes not supported, using regular Adam optimizer')
                optimizer_cls = torch.optim.AdamW
        else:
            optimizer_cls = torch.optim.AdamW
        # L589 just initiates with the settings        

        optimizer = optimizer_cls(
            self.unet.parameters(),             #on haru's this is
            lr=oc.lr,  #args.lr
            betas=(oc.adam_beta1, oc.adam_beta2),                  #args.adam_beta1 (def:0.9), args.adam_beta2 (def:0.999)
            eps=float(oc.adam_epsilon),                    #args.adam_epsilon (def: 1e-08 as float)
            weight_decay=float(oc.adam_weight_decay),            #args.adam_weight_decay (def: 1e-2 as float)
        )

        # Assuming scheduler (on naifus) is the same as lr_scheduler (as harus)
        # Important Note: To not to be confused with noise scheduler! check L646

        #Also known as lr_scheduler
        scheduler = get_scheduler(
            lsc.type,
            optimizer=optimizer,
            num_warmup_steps=int(float(lsc.warmup) * int(self.train_dataloader_len) * int(self.config.trainer.epochs)), #This part is going to be hardcoded until we get a way to get dataset length
            num_training_steps=int(int(self.config.trainer.epochs) * int(self.train_dataloader_len))
        )

        return [[optimizer], [scheduler]]
    
    def on_save_checkpoint(self, checkpoint):
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
        pipeline.save_pretrained(self.output_model_path)
            
    #skipping on_train_start, on_train_batch_end, because they use ema things that I dont use
    #skipping on_load_checkpoint bcuz according to docs we might not use them

def get_class(name: str):
    import importlib

    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)

def load_model(config, train_dataloader_len, tokenizer):
    #the naifu guy downloads the model from a url but im not going to add that
    return StableDiffusionModel(config, train_dataloader_len, tokenizer)