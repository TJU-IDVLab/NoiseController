import logging
import os
import contextlib
from functools import partial
from omegaconf import OmegaConf
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers.optimization import get_scheduler

from ..misc.common import load_module, convert_outputs_to_fp16, move_to
from .multiview_runner import MultiviewRunner
from .base_t_validator import BaseTValidator
from .utils import smart_param_count, prepare_ckpt
from ..networks.unet_2d_condition_multiview import UNet2DConditionModelMultiview
from .noise_multi import noise_slide_window
import numpy as np
from PIL import Image

class MultiviewTRunner(MultiviewRunner):
    def __init__(self, cfg, accelerator, train_set, val_set) -> None:
        super().__init__(cfg, accelerator, train_set, val_set)
        pipe_cls = load_module(cfg.model.pipe_module)
        self.validator = BaseTValidator(
            self.cfg,
            self.val_dataset,
            pipe_cls,
            pipe_param={
                "vae": self.vae,
                "text_encoder": self.text_encoder,
                "tokenizer": self.tokenizer,
            }
        )
        # we set _sc_attn_index here
        if cfg.model.sc_attn_index:
            self._sc_attn_index = OmegaConf.to_container(
                cfg.model.sc_attn_index, resolve=True)
        else:
            self._sc_attn_index = None

    def get_sc_attn_index(self):
        return self._sc_attn_index

    def _init_trainable_models(self, cfg):
        unet = UNet2DConditionModelMultiview.from_pretrained(
            cfg.model.pretrained_magicdrive, subfolder=cfg.model.unet_dir)
        unet_I = UNet2DConditionModelMultiview.from_pretrained(
            cfg.model.pretrained_magicdrive, subfolder=cfg.model.unet_dir)

        model_cls = load_module(cfg.model.unet_module)
        unet_param = OmegaConf.to_container(self.cfg.model.unet, resolve=True)
        unet_I_param = copy.deepcopy(unet_param)
        self.unet = model_cls.from_unet_2d_condition(unet, **unet_param)
        model_cls = load_module(cfg.model.unet_module)
        self.unet_I = model_cls.from_unet_2d_condition(unet_I, **unet_I_param)
        if cfg.model.load_pretrain_from is not None:
            load_path = prepare_ckpt(
                cfg.model.load_pretrain_from,
                self.accelerator.is_local_main_process
            )
            self.accelerator.wait_for_everyone()  # wait
            if cfg.model.allow_partial_load:
                m, u = self.unet.load_state_dict(
                    torch.load(load_path, map_location='cpu'), strict=False)
                logging.info(
                    f"[MultiviewTRunner] weight loaded from {load_path} "
                    f"with missing: {m}, unexpected {u}.")
            else:
                self.unet.load_state_dict(
                    torch.load(load_path, map_location='cpu'))
                self.unet_I.load_state_dict(
                    torch.load(load_path, map_location='cpu'))
                logging.info(
                    f"[MultiviewTRunner] weight loaded from {load_path}")

        model_cls = load_module(cfg.model.model_module)
        controlnet_param = OmegaConf.to_container(
            self.cfg.model.controlnet, resolve=True)
        self.controlnet = model_cls.from_pretrained(
            cfg.model.pretrained_magicdrive, subfolder=cfg.model.controlnet_dir,
            **controlnet_param)

        # add setter func
        for mod in self.unet.modules():
            if hasattr(mod, "_sc_attn_index"):
                mod._sc_attn_index = self.get_sc_attn_index
        for mod in self.unet_I.modules():  
            if hasattr(mod, "_sc_attn_index"):
                mod._sc_attn_index = self.get_sc_attn_index

    def _set_model_trainable_state(self, train=True):
        # set trainable status
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.controlnet.requires_grad_(False)
        self.controlnet.train(False)
        # only unet
        self.unet.requires_grad_(False)
        for name, mod in self.unet.trainable_module.items():
            logging.debug(
                f"[MultiviewRunner] set {name} to requires_grad = True")
            mod.requires_grad_(train)
        self.unet_I.requires_grad_(False)  
        for name, mod in self.unet_I.trainable_module.items():
            logging.debug(
                f"[MultiviewRunner] set {name} to requires_grad = True")
            mod.requires_grad_(train)
 
    def set_optimizer_scheduler(self):
        # optimizer and lr_schedulers
        if self.cfg.runner.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Optimizer creation
        unet_params = self.unet.trainable_parameters
        unet_I_params = self.unet_I.trainable_parameters
        param_count = smart_param_count(unet_params)

        logging.info(
            f"[MultiviewRunner] add {param_count} params from unet to optimizer.")
        
        params_to_optimize = list(unet_params) + list(self.mmodel.parameters()) + list(unet_I_params)
        params_to_optimize.append(self.impact)
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=self.cfg.runner.learning_rate,
            betas=(self.cfg.runner.adam_beta1, self.cfg.runner.adam_beta2),
            weight_decay=self.cfg.runner.adam_weight_decay,
            eps=self.cfg.runner.adam_epsilon,
        )

        # lr scheduler
        self._calculate_steps()

        self.lr_scheduler = get_scheduler(
            self.cfg.runner.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.runner.lr_warmup_steps * self.cfg.runner.gradient_accumulation_steps,
            num_training_steps=self.cfg.runner.max_train_steps * self.cfg.runner.gradient_accumulation_steps,
            num_cycles=self.cfg.runner.lr_num_cycles,
            power=self.cfg.runner.lr_power,
        )


    def prepare_device(self):
        # accelerator
        ddp_modules = (
            self.unet,
            self.unet_I,
            self.mmodel,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.init_w_bi,
            self.impact,
        )
        ddp_modules = self.accelerator.prepare(*ddp_modules)
        (
            self.unet,
            self.unet_I,
            self.mmodel,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.init_w_bi,
            self.impact,
        ) = ddp_modules

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae, unet and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.controlnet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.init_w_bi.to(self.accelerator.device, dtype=self.weight_dtype) 
        self.impact.to(self.accelerator.device, dtype=self.weight_dtype)
        self.mmodel.to(self.accelerator.device, dtype=torch.float32)
        if self.cfg.runner.unet_in_fp16 and self.weight_dtype == torch.float16:
            self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
            self.unet_I.to(self.accelerator.device, dtype=self.weight_dtype)
            # move optimized params to fp32. TODO: is this necessary?
            if self.cfg.model.use_fp32_for_unet_trainable:
                for name, mod in self.accelerator.unwrap_model(
                        self.unet).trainable_module.items():
                    logging.debug(f"[MultiviewRunner] set {name} to fp32")
                    mod.to(dtype=torch.float32)
                    mod._original_forward = mod.forward
                    # autocast intermediate is necessary since others are fp16
                    mod.forward = torch.cuda.amp.autocast(
                        dtype=torch.float16)(mod.forward)
                    # we ensure output is always fp16
                    mod.forward = convert_outputs_to_fp16(mod.forward)

                for name, mod in self.accelerator.unwrap_model(
                        self.unet_I).trainable_module.items():
                    logging.debug(f"[MultiviewRunner] set {name} to fp32")
                    mod.to(dtype=torch.float32)
                    mod._original_forward = mod.forward
                    # autocast intermediate is necessary since others are fp16
                    mod.forward = torch.cuda.amp.autocast(
                        dtype=torch.float16)(mod.forward)
                    # we ensure output is always fp16
                    mod.forward = convert_outputs_to_fp16(mod.forward)
                
                for name, mod in self.accelerator.unwrap_model(self.mmodel).named_modules():
                    mod.to(dtype=torch.float32)
                    mod._original_forward = mod.forward
                    # autocast intermediate is necessary since others are fp16
                    mod.forward = torch.cuda.amp.autocast(
                        dtype=torch.float16)(mod.forward)
                    # we ensure output is always fp16
                    mod.forward = convert_outputs_to_fp16(mod.forward)
            else:
                raise TypeError(
                    "There is an error/bug in accumulation wrapper, please "
                    "make all trainable param in fp32.")

        # no need for this
        self.accelerator.unwrap_model(
            self.controlnet).bbox_embedder._class_tokens_set_or_warned = True

        # We need to recalculate our total training steps as the size of the
        # training dataloader may have changed.
        self._calculate_steps()

    def _save_model(self, root=None):
        if root is None:
            root = self.cfg.log_root

        logging.info(f"w_bi : {self.w_bi*1e2}")
        logging.info(f"impact : {self.impact*1e2}")
        np.savez(f"{os.path.dirname(root)}/{os.path.basename(root).split('_')[0]}/impact.npz",self.impact.detach().cpu().numpy())

    @staticmethod
    def numpy_to_pil(self, images: np.ndarray):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images
    def numpy_to_pil(self, images: np.ndarray):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def decode_latents_super(self, latents, decode_bs=None):
        if decode_bs is not None:
            num_batch = latents.shape[0] // decode_bs
            latents = latents.chunk(num_batch)
            results = []
            for _latents in latents:
                results.append(self.decode_latents(_latents))
            return np.concatenate(results, axis=0)
        else:
            return self.decode_latents(latents)

    def decode_latents(self, latents):
        # decode latents with 5-dims
        latents = 1 / self.vae.config.scaling_factor * latents

        bs = len(latents)
        latents = rearrange(latents, 'b c ... -> (b c) ...')
        image = self.vae.decode(latents).sample
        image = rearrange(image, '(b c) ... -> b c ...', b=bs)

        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = rearrange(image.cpu(), '... c h w -> ... h w c').float().numpy()
        return image

    
    
    def numpy_to_pil_double(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        We need to handle 5-dim inputs and reture 2-dim list.
        """
        imgs_list = []
        for imgs in images:
            imgs_list.append(self.numpy_to_pil(imgs))
        return imgs_list

    def mask_gen(self, coord, x_scale=1, y_scale=1):
        # Calculating mask
        cam_list = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        mask = torch.zeros((16, 6, 224 // x_scale, 400 // y_scale), device=device)
        for fir,frame in enumerate(coord):
            for sec, cam in enumerate(cam_list): 
                coords = frame[cam] 
                for zuobiao in coords: 
                    if len(zuobiao)==0:
                        continue
                    for index in range(len(zuobiao)):
                        x_left = int(zuobiao[index][0] / x_scale) 
                        y_left = int(zuobiao[index][1] / y_scale) 
                        x_right = int(zuobiao[index][2] / x_scale) 
                        y_right = int(zuobiao[index][3] / y_scale) 
                        mask[fir,sec, y_left:y_right, x_left:x_right] = 1 
        return mask

    def _train_one_step(self, batch, residual_proportion):
        torch.cuda.empty_cache()
        self.unet.train()
        self.unet_I.train()

        with self.accelerator.accumulate(self.unet), self.accelerator.accumulate(self.unet_I):
            N_frame = batch["pixel_values"].shape[1]
            N_cam = batch["pixel_values"].shape[2]
            # Convert images to latent space
            latents = self.vae.encode(
                rearrange(
                    batch["pixel_values"],
                    "b l n c h w -> (b l n) c h w").to(
                    dtype=self.weight_dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            latents = rearrange(
                latents, "(b l n) c h w -> b l n c h w", l=N_frame, n=N_cam)
            channel = latents.shape[3]
            height = latents.shape[4]
            width = latents.shape[5]
            # embed camera params, in (B, 6, 3, 7), out (B, 6, 189)

            camera_param = batch["camera_param"].to(self.weight_dtype)

            # Sample noise that we'll add to the latents
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
            p_B_w_impact_loss = torch.tensor(0., device=device).requires_grad_(True)
            p_I_w_impact_loss = torch.tensor(0., device=device).requires_grad_(True)
            self.w_bi = self.mmodel(self.init_w_bi)
            # Calculate the noise and share ratio by sliding window
            noise_B,noise_I,B_w_impact_loss, I_w_impact_loss = \
                noise_slide_window(
                    total_w=self.w_bi , 
                    total_impact=self.impact, 
                    B_w_impact_loss=p_B_w_impact_loss, 
                    I_w_impact_loss=p_I_w_impact_loss, 
                    h=height,
                    w=width,
                    c=channel,
                    batch_size=batch["pixel_values"].shape[0],
                    residual_proportion=residual_proportion
                )
            noise_B = noise_B.to(self.accelerator.device,self.weight_dtype)
            noise_I = noise_I.to(self.accelerator.device,self.weight_dtype)
            B_w_impact_loss = B_w_impact_loss.to(self.accelerator.device,self.weight_dtype)
            I_w_impact_loss = I_w_impact_loss.to(self.accelerator.device,self.weight_dtype)
            
            # make sure we use same noise for different views, only take the
            # first
            if self.cfg.model.train_with_same_noise:
                noise_B = repeat(noise_B[:, :, 1], "b l ... -> b l r ...", r=N_cam)
                noise_I = repeat(noise_I[:, :, 1], "b l ... -> b l r ...", r=N_cam)
            if self.cfg.model.train_with_same_noise_t:
                noise_B = repeat(noise_B[:, 0], "b ... -> b r ...", r=N_frame)
                noise_I = repeat(noise_I[:, 0], "b ... -> b r ...", r=N_frame)

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            assert self.cfg.model.train_with_same_t
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            # add frame dim
            timesteps = repeat(timesteps, "b ... -> b r ...", r=N_frame)
            timesteps = timesteps.long()

            #### here we change (b, l, ...) to (bxl, ...) ####
            f_to_b = partial(rearrange, pattern="b l ... -> (b l) ...")
            b_to_f = partial(
                rearrange, pattern="(b l) ... -> b l ...", l=N_frame)
            latents = f_to_b(latents)
            noise_B = f_to_b(noise_B)
            noise_I = f_to_b(noise_I)
            timesteps = f_to_b(timesteps)
            camera_param = f_to_b(camera_param)
            if batch['kwargs']['bboxes_3d_data'] is not None:
                batch_kwargs = {
                    "bboxes_3d_data": {
                        'bboxes': f_to_b(batch['kwargs']['bboxes_3d_data']['bboxes']),
                        'classes': f_to_b(batch['kwargs']['bboxes_3d_data']['classes']),
                        'masks': f_to_b(batch['kwargs']['bboxes_3d_data']['masks']),
                    }
                }
            else:
                batch_kwargs = {"bboxes_3d_data": None}

            latents_mask = self.mask_gen(batch['meta_data']['coord'][0], 8, 8).half()[:noise_B.shape[0], ...]
            latents_mask = latents_mask.unsqueeze(2)
            latents_mask_ = 1 - latents_mask

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)

            noise_B_mask = noise_B * latents_mask_
            noise_I_mask = noise_I * latents_mask
            noise_B_I = noise_B_mask + noise_I_mask

            noisy_latents = self._add_noise(latents, noise_B_I ,timesteps)
            
            #### here we change (b, l, ...) to (bxl, ...) ####
            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(
                f_to_b(batch["input_ids"]))[0]
            encoder_hidden_states_uncond = self.text_encoder(
                f_to_b(batch["uncond_ids"]))[0]

            controlnet_image = batch["bev_map_with_aux"].to(
                dtype=self.weight_dtype)
            controlnet_image = f_to_b(controlnet_image)
            # fmt: off
            down_block_res_samples, mid_block_res_sample, \
            encoder_hidden_states_with_cam = self.controlnet(
                noisy_latents,  
                timesteps,  
                camera_param=camera_param,  
                encoder_hidden_states=encoder_hidden_states, 
                encoder_hidden_states_uncond=encoder_hidden_states_uncond, 
                controlnet_cond=controlnet_image,  
                return_dict=False,
                **batch_kwargs,
            )
            # fmt: on

            # starting from here, we use (B n) as batch_size
            noisy_latents = rearrange(noisy_latents, "b n ... -> (b n) ...")
            if timesteps.ndim == 1:
                timesteps = repeat(timesteps, "b -> (b n)", n=N_cam)

            # Predict the noise residual
            # NOTE: Since we fix most of the model, we cast the model to fp16 and
            # disable autocast to prevent it from falling back to fp32. Please
            # enable autocast on your customized/trainable modules.
            context = contextlib.nullcontext
            context_kwargs = {}
            if self.cfg.runner.unet_in_fp16:
                context = torch.cuda.amp.autocast
                context_kwargs = {"enabled": False}
            with context(**context_kwargs):
                model_pred = self.unet(
                    noisy_latents,  
                    timesteps.reshape(-1),  
                    encoder_hidden_states=encoder_hidden_states_with_cam.to(
                        dtype=self.weight_dtype
                    ),  
                    # TODO: during training, some camera param are masked.
                    down_block_additional_residuals=[
                        sample.to(dtype=self.weight_dtype)
                        for sample in down_block_res_samples
                    ],  # all intermedite have four dims: b x n, c, h, w
                    mid_block_additional_residual=mid_block_res_sample.to(
                        dtype=self.weight_dtype
                    ),  # b x n, 1280, h, w. we have 4 x 7 as mid_block_res
                ).sample
                
                model_pred_I = self.unet_I(
                    noisy_latents, 
                    timesteps.reshape(-1),  
                    encoder_hidden_states=encoder_hidden_states_with_cam.to(
                        dtype=self.weight_dtype
                    ),  
                    # TODO: during training, some camera param are masked.
                    down_block_additional_residuals=[
                        sample.to(dtype=self.weight_dtype)
                        for sample in down_block_res_samples
                    ],  # all intermedite have four dims: b x n, c, h, w
                    mid_block_additional_residual=mid_block_res_sample.to(
                        dtype=self.weight_dtype
                    ),  # b x n, 1280, h, w. we have 4 x 7 as mid_block_res
                ).sample


            model_pred = rearrange(model_pred, "(b n) ... -> b n ...", n=N_cam)
            model_pred_I = rearrange(model_pred_I, "(b n) ... -> b n ...", n=N_cam)
            
            #### change dims back ####
            noise_I = b_to_f(noise_I)
            noise_B = b_to_f(noise_B)
            noise_B_mask = b_to_f(noise_B_mask)
            noise_I_mask = b_to_f(noise_I_mask)
            model_pred = b_to_f(model_pred)
            moedl_pred_I = b_to_f(model_pred_I)
            
            # the loss of image
            loss_b = F.mse_loss(
                (model_pred * latents_mask_).float(), noise_B_mask.float(), reduction='none')
            loss_i = F.mse_loss(
                (moedl_pred_I * latents_mask).float(), noise_I_mask.float(), reduction='none') 

            # the loss of share ratio
            loss_w_impact_B = F.l1_loss(
                B_w_impact_loss.float(), torch.empty(B_w_impact_loss.shape, device=B_w_impact_loss.device).fill_(1.-residual_proportion).float(), reduction='none'
            )
            loss_w_impact_I = F.l1_loss(
                I_w_impact_loss.float(), torch.empty(I_w_impact_loss.shape, device=I_w_impact_loss.device).fill_(1.-residual_proportion).float(), reduction='none'
            )
            
            loss = loss_b.mean() \
                + loss_i.mean() \
                + loss_w_impact_B.mean() \
                + loss_w_impact_I.mean()

            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients and self.cfg.runner.max_grad_norm is not None:
                params_to_clip = list(self.unet.parameters())+ list(self.mmodel.parameters()) + list(self.unet_I.parameters())
                self.accelerator.clip_grad_norm_(
                    params_to_clip, self.cfg.runner.max_grad_norm
                )

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(
                set_to_none=self.cfg.runner.set_grads_to_none)

        return loss, tensorboard_logs
