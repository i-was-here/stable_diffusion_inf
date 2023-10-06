import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
model_id = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
out_img = Image.new("RGB", (768, 768), color="black")
pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                               scheduler=scheduler,
                                               vae=vae,
                                              #  safety_checker=safe,
                                               revision="fp16",
                                               torch_dtype=torch.float16)
pipe.enable_xformers_memory_efficient_attention()
pipe = pipe.to("cuda")
def prompt_to_image(prompt):
  results = pipe(prompt, height=768, width=768, guidance_scale = 10)
  if not results.nsfw_content_detected[0]: return results.images[0]
  else: return out_img
import gradio as gr
gr.Interface(prompt_to_image, gr.Text(), gr.Image(), title = 'Stable Diffusion 2.0 Colab with Gradio UI').launch(share = True, debug = True)
