import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
out_img = Image.new("RGB", (768, 768), color="black")
out_img.info.update({'contest': 'H4r61y Hum4n5 c0n73s7 1'})
def prompt_to_image(prompt):
  model_id = "stabilityai/stable-diffusion-2"
  scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", torch_dtype=torch.float16)
  vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
  pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                 scheduler=scheduler,
                                                 vae=vae,
                                                #  safety_checker=safe,
                                                 revision="fp16",
                                                 torch_dtype=torch.float16)
  pipe.enable_xformers_memory_efficient_attention()
  pipe = pipe.to("cuda")
  results = pipe(prompt, height=768, width=768, guidance_scale = 10)
  results.images[0].info.update({'contest': 'H4r61y Hum4n5 c0n73s7 1'})
  if not results.nsfw_content_detected[0]: return results.images[0]
  else: return out_img.info.update({'contest': 'H4r61y Hum4n5 c0n73s7 1'})

def get_image(res):
  if not res.nsfw_content_detected[0]:
    res.images[0].info.update({'contest': 'H4r61y Hum4n5 c0n73s7 1'})
    return res.images[0]
  else: return out_img

def extract_process(original_image, watermark_text, position=(0, 0), font_size=400, opacity=100):
  watermarked_image = original_image.convert("RGBA").copy()

  try:
      font = ImageFont.truetype("arial.ttf", font_size)
  except IOError:
      font = ImageFont.load_default()

  draw = ImageDraw.Draw(watermarked_image, "RGBA")

  text_width, text_height = draw.textsize(watermark_text, font)
  width, height = watermarked_image.size
  x = (width - text_width) / 2
  y = (height - text_height) / 2

  overlay = Image.new("RGBA", watermarked_image.size)
  overlay_draw = ImageDraw.Draw(overlay, "RGBA")
  overlay_draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, opacity))

  watermarked_image = Image.alpha_composite(watermarked_image, overlay)

  return watermarked_image

import gradio as gr

if __name__=="__main__":
  gr.Interface(prompt_to_image, gr.Text(), gr.Image(), title = 'Stable Diffusion 2.0 Colab with Gradio UI').launch(share = True, debug = True)
