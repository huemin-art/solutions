import torch
# from diffusers import FluxKontextPipeline
from pipeline import FluxKontextPipeline
from diffusers.utils import load_image

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

image = load_image("input.png")
original_width, original_height = image.size
image.save("img0.png")

for i in range(5):
  # resize the image to 1328*800
  image = image.resize((1328, 800))
  width, height = image.size
  print(f"Image size: {width}x{height}")

  # do diffusion 
  image = pipe(
    image=image,
    height=height,
    width=width,
    prompt="do nothing to the image",
    guidance_scale=1.0,
    num_inference_steps=20,
    max_area=width*height,
  ).images[0]

  # resize back to original size
  image = image.resize((original_width, original_height))
  image.save(f"img{i+1}.png")