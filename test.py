import torch
# from diffusers import FluxKontextPipeline
from pipeline import FluxKontextPipeline
from diffusers.utils import load_image

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

input_image = load_image("input.png")

image = pipe(
  image=input_image,
  width=1280, # same as input image
  height=720, # same as input image
  prompt="add a unicorn to the image",
  guidance_scale=1.5
).images[0]

image.save("output_good.png")
