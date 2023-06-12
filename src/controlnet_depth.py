from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from huggingface_hub import hf_hub_download
from transformers import CLIPTextModel, CLIPTokenizer, logging

# suppress partial model loading warning
from src import utils
from src.utils import seed_everything
from src.annotator.util import resize_image, HWC3
from src.cldm.model import create_model, load_state_dict

logging.set_verbosity_error()
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm.auto import tqdm
import cv2
import numpy as np
from PIL import Image
import einops


class ControlNet(nn.Module):
    def __init__(self, device, model_name='CompVis/stable-diffusion-v1-4', concept_name=None, concept_path=None,
                 latent_mode=True,  min_timestep=0.02, max_timestep=0.98, no_noise=False,
                 use_inpaint=False):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '')  # remove the last \n!
                logger.info(f'loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            logger.warning(
                f'try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')

        self.device = device
        self.latent_mode = latent_mode
        self.no_noise = no_noise
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * min_timestep)
        self.max_step = int(self.num_train_timesteps * max_timestep)
        self.use_inpaint = use_inpaint

        logger.info(f'loading stable diffusion with {model_name}...')

        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", use_auth_token=self.token).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer', use_auth_token=self.token)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder',
                                                          use_auth_token=self.token).to(self.device)
        self.image_encoder = None
        self.image_processor = None

        # 3. The UNet model for generating the latents.
        # CHANGED TO REPLACE UNET WITH CONTROLNET
        logger.info(f'loading control_net from ./src/models/cldm_v15.yaml...')
        self.unet = create_model('./src/models/cldm_v15.yaml').cpu()
        self.unet.load_state_dict(load_state_dict('./src/models/control_sd15_depth.pth', location=self.device))
        self.unet.to(self.device)

        if self.use_inpaint:
            self.inpaint_unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting",
                                                                     subfolder="unet", use_auth_token=self.token).to(
                self.device)


        # 4. Create a scheduler for inference
        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                       num_train_timesteps=self.num_train_timesteps, steps_offset=1,
                                       skip_prk_steps=True)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) 

        if concept_name is not None:
            self.load_concept(concept_name, concept_path)
        logger.info(f'\t successfully loaded stable diffusion!')

    def load_concept(self, concept_name, concept_path=None):
        if concept_path is None:
            repo_id_embeds = f"sd-concepts-library/{concept_name}"
            learned_embeds_path = hf_hub_download(repo_id=repo_id_embeds, filename="learned_embeds.bin")
        else:
            learned_embeds_path = concept_path

        loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

        # separate token and the embeds
        for trained_token in loaded_learned_embeds:
            # trained_token = list(loaded_learned_embeds.keys())[0]
            print(f'Loading token for {trained_token}')
            embeds = loaded_learned_embeds[trained_token]

            # cast to dtype of text_encoder
            dtype = self.text_encoder.get_input_embeddings().weight.dtype
            embeds.to(dtype)

            # add the token in tokenizer
            token = trained_token
            num_added_tokens = self.tokenizer.add_tokens(token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

            # resize the token embeddings
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))

            # get the id for the token and assign the embeds
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            self.text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    def get_text_embeds(self, prompt, negative_prompt=None):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        logger.info(prompt)
        logger.info(text_input.input_ids)

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        if negative_prompt is None:
            negative_prompt = [''] * len(prompt)
        uncond_input = self.tokenizer(negative_prompt, padding='max_length',
                                      max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    
    def get_control_net_inputs(self, depth_mask, prompt, a_prompt, n_prompt, image_resolution, strength):
        num_samples = 1
        guess_mode = False
        input_image = np.uint8(np.array(depth_mask[0,0,:,:].cpu())*255)
        input_image = HWC3(input_image)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        input_image = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(input_image.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        cond = {"c_concat": [control], "c_crossattn": [self.unet.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.unet.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        self.unet.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        return cond, un_cond



    def img2img_step(self, text_embeddings, inputs, depth_mask, control_net_cond, control_net_uncond,
                     guidance_scale=100, strength=0.5, num_inference_steps=50, update_mask=None, 
                     latent_mode=False, check_mask=None, fixed_seed=None, check_mask_iters=0.5, intermediate_vis=False):
        # input is 1 3 512 512
        # depth_mask is 1 1 512 512
        # text_embeddings is 2 512
        intermediate_results = []

        def sample(latents, depth_mask, strength, num_inference_steps, update_mask=None, check_mask=None,
                   masked_latents=None):
            self.scheduler.set_timesteps(num_inference_steps)
            noise = None
            if latents is None:
                # Last chanel is reserved for depth 
                latents = torch.randn(
                    (
                        text_embeddings.shape[0] // 2, self.unet.control_model.in_channels, depth_mask.shape[2],
                        depth_mask.shape[3]),
                    device=self.device)
                timesteps = self.scheduler.timesteps
            else:
                # Strength has meaning only when latents are given
                timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
                latent_timestep = timesteps[:1]
                if fixed_seed is not None:
                    seed_everything(fixed_seed)
                noise = torch.randn_like(latents)
                if update_mask is not None:
                    gt_latents = latents
                    latents = torch.randn(
                        (text_embeddings.shape[0] // 2, self.unet.control_model.in_channels, depth_mask.shape[2],
                         depth_mask.shape[3]),
                        device=self.device)
                else:
                    latents = self.scheduler.add_noise(latents, noise, latent_timestep)

            depth_mask = torch.cat([depth_mask] * 2)

            with torch.autocast('cuda'):
                for i, t in tqdm(enumerate(timesteps)):
                    is_inpaint_range = self.use_inpaint and (10 < i < 20)
                    mask_constraints_iters = True  # i < 20
                    is_inpaint_iter = is_inpaint_range  # and i %2 == 1

                    if not is_inpaint_range and mask_constraints_iters:
                        if update_mask is not None:
                            noised_truth = self.scheduler.add_noise(gt_latents, noise, t)
                            if check_mask is not None and i < int(len(timesteps) * check_mask_iters):
                                curr_mask = check_mask
                            else:
                                curr_mask = update_mask
                            latents = latents * curr_mask + noised_truth * (1 - curr_mask)

                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input,
                                                                          t)  # NOTE: This does nothing

                    if is_inpaint_iter:
                        latent_mask = torch.cat([update_mask] * 2)
                        latent_image = torch.cat([masked_latents] * 2)
                        latent_model_input_inpaint = torch.cat([latent_model_input, latent_mask, latent_image], dim=1)
                        with torch.no_grad():
                            noise_pred_inpaint = \
                                self.inpaint_unet(latent_model_input_inpaint, t, encoder_hidden_states=text_embeddings)[
                                    'sample']
                            noise_pred = noise_pred_inpaint
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    else:
                        # CHANGED TO USE CONTROLNET
                        # predict the noise residual
                        with torch.no_grad():
                            ts = torch.full((1,), t, device=self.device, dtype=torch.long)
                            noise_pred_text = self.unet.apply_model(latents, ts, control_net_cond)
                            noise_pred_uncond = self.unet.apply_model(latents, ts, control_net_uncond)

                    # perform guidance
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1

                    if intermediate_vis:
                        vis_alpha_t = torch.sqrt(self.scheduler.alphas_cumprod)
                        vis_sigma_t = torch.sqrt(1 - self.scheduler.alphas_cumprod)
                        a_t, s_t = vis_alpha_t[t], vis_sigma_t[t]
                        vis_latents = (latents - s_t * noise) / a_t
                        vis_latents = 1 / 0.18215 * vis_latents
                        image = self.vae.decode(vis_latents).sample
                        image = (image / 2 + 0.5).clamp(0, 1)
                        image = image.cpu().permute(0, 2, 3, 1).numpy()
                        image = Image.fromarray((image[0] * 255).round().astype("uint8"))
                        intermediate_results.append(image)
                    latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

            return latents

        depth_mask = F.interpolate(depth_mask, size=(64, 64), mode='bicubic',
                                   align_corners=False)
        masked_latents = None
        if inputs is None:
            latents = None
        elif latent_mode:
            latents = inputs
        else:
            pred_rgb_512 = F.interpolate(inputs, (512, 512), mode='bilinear',
                                         align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)
            if self.use_inpaint:
                update_mask_512 = F.interpolate(update_mask, (512, 512))
                masked_inputs = pred_rgb_512 * (update_mask_512 < 0.5) + 0.5 * (update_mask_512 >= 0.5)
                masked_latents = self.encode_imgs(masked_inputs)

        if update_mask is not None:
            update_mask = F.interpolate(update_mask, (64, 64), mode='nearest')
        if check_mask is not None:
            check_mask = F.interpolate(check_mask, (64, 64), mode='nearest')

        depth_mask = 2.0 * (depth_mask - depth_mask.min()) / (depth_mask.max() - depth_mask.min()) - 1.0

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = (self.min_step + self.max_step) // 2

        with torch.no_grad():
            target_latents = sample(latents, depth_mask, strength=strength, num_inference_steps=num_inference_steps,
                                    update_mask=update_mask, check_mask=check_mask, masked_latents=masked_latents)
            target_rgb = self.decode_latents(target_latents)

        if latent_mode:
            return target_rgb, target_latents
        else:
            return target_rgb, intermediate_results

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

