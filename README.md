# DualGuide-SDXL

**Dual Guidance for Stable Diffusion XL: Token Perturbation Guidance + Normalized Attention Guidance**

A powerful, training-free enhancement for Stable Diffusion XL that combines **Token Perturbation Guidance (TPG)** and **Normalized Attention Guidance (NAG)** to dramatically improve image generation quality and prompt adherence.

## Overview

This project implements two advanced guidance techniques for Stable Diffusion XL:

1. **Token Perturbation Guidance (TPG)**: A training-free method that improves generation quality by perturbing token sequences during inference and using the difference as guidance signal.

2. **Normalized Attention Guidance (NAG)**: Enhances attention mechanisms by normalizing attention maps and extrapolating conditional attention, leading to better prompt adherence and image quality.

## Features

- **Training-Free**: No additional training required, works with any SDXL checkpoint
- **Flexible Layer Selection**: Apply TPG to specific transformer layers for fine-grained control
- **NAG Support**: Optional normalized attention guidance for improved results
- **Compatible**: Works seamlessly with LoRA, textual inversion, and other Diffusers features
- **Easy Integration**: Drop-in replacement for standard SDXL pipeline

## Project Structure

```
DualGuide-SDXL/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── src/                          # Source code
│   ├── __init__.py
│   ├── pipelines/                # Pipeline implementations
│   │   ├── __init__.py
│   │   └── pipeline_sdxl_tpg.py  # TPG pipeline for SDXL
│   └── processors/               # Attention processors
│       ├── __init__.py
│       └── custom_attn.py        # NAG and SoftPAG processors
├── examples/                     # Usage examples
│   ├── __init__.py
│   └── basic_usage.py            # Basic usage example
└── outputs/                      # Generated images
    └── sdxl_output.png
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 12GB+ VRAM for SDXL)
- PyTorch 2.0+

### Setup

1. Clone this repository:
```bash
git clone https://github.com/v0ntr3n/tpg_nag.git
cd tpg_nag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional but recommended) Install xformers for better memory efficiency:
```bash
pip install xformers
```

## Quick Start

### Basic Usage

```python
import torch
from diffusers import AutoencoderKL
from src.pipelines.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline

# Load the pipeline
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

# Generate an image with TPG and NAG
prompt = "a beautiful landscape with mountains and a lake, dramatic lighting"

image = pipe(
    prompt,
    width=1024,
    height=1024,
    num_inference_steps=25,
    guidance_scale=7.5,     # CFG guidance scale
    tpg_scale=3.0,          # TPG guidance strength
    tpg_applied_layers_index=[
        "d6", "d7", "d8", "d9", "d10", "d11",
        "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23"
    ],
    do_nag=True,            # Enable NAG
    nag_alpha=1.0,          # NAG strength
    nag_use_l1=True         # Use L1 normalization
).images[0]

image.save("output.png")
```

### Using Normalized Attention Guidance (NAG)

NAG is integrated directly into the pipeline. Simply enable it with the `do_nag` parameter:

```python
# Generate with both TPG and NAG
image = pipe(
    prompt,
    width=1024,
    height=1024,
    num_inference_steps=25,
    guidance_scale=7.5,
    tpg_scale=3.0,
    do_nag=True,        # Enable NAG
    nag_alpha=1.0,      # NAG extrapolation strength
    nag_use_l1=True     # Use L1 normalization
).images[0]
```

**Note**: The `apply_nag()` function in `custom_attn.py` is also available if you need to manually apply NAG processors to the UNet, but it's not needed when using the integrated pipeline parameters.

## Parameters

### Token Perturbation Guidance Parameters

- `tpg_scale` (float, default: 3.0): Strength of token perturbation guidance. Higher values increase the effect.
- `tpg_applied_layers_index` (list): List of layer indices to apply TPG. Format: `["d6", "d7", ...]` for down layers, `["m0"]` for mid layers, `["u0", "u1", ...]` for up layers.

### Normalized Attention Guidance Parameters

- `do_nag` (bool, default: False): Enable NAG processors.
- `nag_alpha` (float, default: 1.0): NAG extrapolation strength.
- `nag_use_l1` (bool, default: True): Use L1 normalization for attention maps.

### Standard SDXL Parameters

- `guidance_scale` (float, default: 7.5): Classifier-free guidance scale.
- `num_inference_steps` (int, default: 50): Number of denoising steps.
- `negative_prompt` (str): Negative prompt for guidance.

## Advanced Usage

### Different Generation Modes

The pipeline supports three guidance modes that can be used individually or combined:

```python
# 1. Standard CFG only (no TPG, no NAG)
image = pipe(
    prompt,
    guidance_scale=7.5,
    tpg_scale=0.0  # Disable TPG
).images[0]

# 2. TPG only (no CFG)
image = pipe(
    prompt,
    guidance_scale=0.0,  # Disable CFG
    tpg_scale=3.0,
    tpg_applied_layers_index=["d6", "d7", "d8", "d9", "d10"]
).images[0]

# 3. CFG + TPG + NAG (all combined)
image = pipe(
    prompt,
    guidance_scale=7.5,  # CFG enabled
    tpg_scale=3.0,       # TPG enabled
    tpg_applied_layers_index=["d6", "d7", "d8", "d9", "d10"],
    do_nag=True,         # NAG enabled
    nag_alpha=1.0
).images[0]
```

### Layer Selection for TPG

The pipeline supports fine-grained control over which transformer layers use TPG:

```python
# Apply TPG only to specific layers
layers = [
    "d10", "d11", "d12",  # Down layers 10-12
    "m0",                  # Mid layer 0
    "u0", "u1", "u2"      # Up layers 0-2
]

image = pipe(
    prompt,
    tpg_scale=3.0,
    tpg_applied_layers_index=layers
).images[0]
```

### Combining with LoRA

```python
# Load LoRA weights
pipe.load_lora_weights("path/to/lora_weights.safetensors")

# Generate with both LoRA and TPG
image = pipe(
    prompt,
    tpg_scale=3.0,
    guidance_scale=7.5
).images[0]
```

### Using SoftPAG (Soft Perturbed Attention Guidance)

```python
from src.processors.custom_attn import apply_softpag

# Apply SoftPAG to selected attention heads
pipe = apply_softpag(
    pipe,
    beta=0.5,              # Interpolation factor toward identity
    selected_heads=[0, 1]  # Apply to specific heads (None = all heads)
)

image = pipe(prompt).images[0]
```

## Technical Details

### Token Perturbation Guidance

TPG works by:
1. Creating a perturbed version of the token sequence by randomly shuffling token positions
2. Running both original and perturbed sequences through selected transformer layers
3. Using the difference between predictions as an additional guidance signal
4. Combining with standard classifier-free guidance (if enabled)

The guidance formula with both CFG and TPG is:
```
noise_pred = noise_text + (cfg_scale - 1.0) * (noise_text - noise_uncond) + tpg_scale * (noise_text - noise_perturbed)
```

### Normalized Attention Guidance

NAG enhances attention by:
1. Computing attention maps for conditional and unconditional branches
2. Applying L1 normalization to attention weights
3. Extrapolating conditional attention using the difference from unconditional
4. Renormalizing to maintain valid attention distributions

## Performance Tips

1. **Memory Usage**:
   - Use `torch.float16` for reduced VRAM usage
   - Install xformers for efficient attention computation
   - Reduce image resolution or batch size if needed

2. **Quality vs Speed**:
   - More inference steps → better quality but slower
   - Higher TPG scale → stronger effect but may oversaturate
   - Apply TPG to fewer layers for faster inference

3. **Layer Selection**:
   - Down layers (d6-d23): Affect high-level structure
   - Mid layers (m0): Affect overall composition
   - Up layers (u0-u23): Affect fine details

## Citation

If you use this code in your research, please cite the relevant papers:

```bibtex
@article{tpg2025,
  title={Token Perturbation Guidance for Diffusion Models},
  journal={arXiv:2506.10036},
  year={2025}
}

@article{nag2025,
  title={Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models},
  journal={arXiv:2505.21179},
  year={2025}
}
```

## License

This project is provided for research and educational purposes. Please check the licenses of the underlying models (Stable Diffusion XL) and libraries (Diffusers, Transformers) for commercial use.

## Acknowledgments

- Built on [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- Uses [Stable Diffusion XL](https://stability.ai/stable-diffusion) by Stability AI
- Inspired by recent advances in guidance techniques for diffusion models

## Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce image resolution (try 512x512 instead of 1024x1024)
   - Use fewer TPG layers
   - Enable model offloading: `pipe.enable_model_cpu_offload()`

2. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path includes the project root

3. **Quality Issues**:
   - Adjust `tpg_scale` (try values between 2.0 and 5.0)
   - Experiment with different layer selections
   - Increase `num_inference_steps` for better quality

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or issues, please open an issue on the GitHub repository.
