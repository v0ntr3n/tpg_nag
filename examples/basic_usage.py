"""
Basic usage example for TPG-NAG pipeline.

This script demonstrates the simplest way to use the TPG pipeline
with both Token Perturbation Guidance and Normalized Attention Guidance.
"""

import time
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline

# Load the pipeline with optimized VAE
print("Loading pipeline...")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLTPGPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

# Optional: Load LoRA weights if available
# pipe.load_lora_weights('pytorch_lora_weights.safetensors')

# Your prompt
prompt = (
    "a bathroom with a marble bathtub and marble countertop. "
    "The bathtub is surrounded by three windows, providing natural light into the room. "
    "The bathroom also features a sink situated near the countertop. "
    "In addition to the bathroom fixtures, there is a vase placed on the countertop"
)

# Generate image with TPG and NAG
print("\nGenerating image with TPG + NAG...")
start_time = time.time()

output = pipe(
    prompt,
    width=512,
    height=512,
    num_inference_steps=25,
    guidance_scale=7.5,      # CFG scale
    tpg_scale=3.0,           # TPG scale (set to 0.0 to disable)
    tpg_applied_layers_index=[
        "d6", "d7", "d8", "d9", "d10", "d11",
        "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23"
    ],
    do_nag=True,             # Enable NAG
    nag_alpha=1.0,           # NAG strength
    nag_use_l1=True          # Use L1 normalization
).images[0]

# Save the output
output_dir = Path(__file__).parent.parent / "outputs"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'basic_usage_output.png'
output.save(output_path)

end_time = time.time()
print(f"\nImage generated in {end_time - start_time:.2f} seconds")
print(f"Saved to: {output_path}")