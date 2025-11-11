"""
Advanced usage examples for TPG-NAG pipeline.

This script demonstrates:
1. Different guidance modes (CFG only, TPG only, combined)
2. NAG integration
3. Layer selection strategies
4. Performance optimization
"""

import sys
from pathlib import Path
import time

import torch
from diffusers import AutoencoderKL

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline


def setup_pipeline():
    """Initialize the pipeline with optimized settings."""
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

    return pipe


def example_cfg_only(pipe, prompt, output_dir):
    """Example 1: Standard CFG only (baseline)."""
    print("\n=== Example 1: CFG Only (Baseline) ===")
    start = time.time()

    image = pipe(
        prompt,
        width=512,
        height=512,
        num_inference_steps=25,
        guidance_scale=7.5,
        tpg_scale=0.0  # Disable TPG
    ).images[0]

    elapsed = time.time() - start
    image.save(output_dir / "example_cfg_only.png")
    print(f"Generated in {elapsed:.2f}s")


def example_tpg_only(pipe, prompt, output_dir):
    """Example 2: TPG only (no CFG)."""
    print("\n=== Example 2: TPG Only ===")
    start = time.time()

    image = pipe(
        prompt,
        width=512,
        height=512,
        num_inference_steps=25,
        guidance_scale=0.0,  # Disable CFG
        tpg_scale=3.0,
        tpg_applied_layers_index=[
            "d6", "d7", "d8", "d9", "d10", "d11",
            "d12", "d13", "d14", "d15", "d16", "d17"
        ]
    ).images[0]

    elapsed = time.time() - start
    image.save(output_dir / "example_tpg_only.png")
    print(f"Generated in {elapsed:.2f}s")


def example_combined(pipe, prompt, output_dir):
    """Example 3: CFG + TPG combined."""
    print("\n=== Example 3: CFG + TPG Combined ===")
    start = time.time()

    image = pipe(
        prompt,
        width=512,
        height=512,
        num_inference_steps=25,
        guidance_scale=7.5,  # CFG enabled
        tpg_scale=3.0,       # TPG enabled
        tpg_applied_layers_index=[
            "d6", "d7", "d8", "d9", "d10", "d11",
            "d12", "d13", "d14", "d15", "d16", "d17",
            "d18", "d19", "d20", "d21", "d22", "d23"
        ]
    ).images[0]

    elapsed = time.time() - start
    image.save(output_dir / "example_combined.png")
    print(f"Generated in {elapsed:.2f}s")


def example_with_nag(pipe, prompt, output_dir):
    """Example 4: CFG + TPG + NAG."""
    print("\n=== Example 4: CFG + TPG + NAG ===")
    start = time.time()

    image = pipe(
        prompt,
        width=512,
        height=512,
        num_inference_steps=25,
        guidance_scale=7.5,
        tpg_scale=3.0,
        tpg_applied_layers_index=[
            "d6", "d7", "d8", "d9", "d10", "d11",
            "d12", "d13", "d14", "d15", "d16", "d17",
            "d18", "d19", "d20", "d21", "d22", "d23"
        ],
        do_nag=True,        # Enable NAG
        nag_alpha=1.0,      # NAG strength
        nag_use_l1=True     # Use L1 normalization
    ).images[0]

    elapsed = time.time() - start
    image.save(output_dir / "example_with_nag.png")
    print(f"Generated in {elapsed:.2f}s")


def example_layer_strategies(pipe, prompt, output_dir):
    """Example 5: Different layer selection strategies."""
    print("\n=== Example 5: Layer Selection Strategies ===")

    strategies = {
        "deep_layers": ["d18", "d19", "d20", "d21", "d22", "d23"],
        "mid_layers": ["d10", "d11", "d12", "d13", "d14", "d15"],
        "shallow_layers": ["d6", "d7", "d8", "d9"],
        "mid_block": ["m0"],
        "mixed": ["d6", "d10", "d15", "m0", "u5", "u10"]
    }

    for strategy_name, layers in strategies.items():
        print(f"\nTesting strategy: {strategy_name}")
        start = time.time()

        image = pipe(
            prompt,
            width=512,
            height=512,
            num_inference_steps=20,
            guidance_scale=7.5,
            tpg_scale=3.0,
            tpg_applied_layers_index=layers
        ).images[0]

        elapsed = time.time() - start
        image.save(output_dir / f"example_layers_{strategy_name}.png")
        print(f"  Generated in {elapsed:.2f}s with layers: {layers}")


def main():
    """Run all examples."""
    # Setup
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    prompt = (
        "a serene Japanese garden with a wooden bridge over a koi pond, "
        "cherry blossoms falling, soft morning light, highly detailed, "
        "professional photography, 8k"
    )

    print("Initializing pipeline...")
    pipe = setup_pipeline()

    # Run examples
    example_cfg_only(pipe, prompt, output_dir)
    example_tpg_only(pipe, prompt, output_dir)
    example_combined(pipe, prompt, output_dir)
    example_with_nag(pipe, prompt, output_dir)
    example_layer_strategies(pipe, prompt, output_dir)

    print("\n=== All examples completed! ===")
    print(f"Images saved to: {output_dir}")


if __name__ == "__main__":
    main()
