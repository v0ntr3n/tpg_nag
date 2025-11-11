"""DualGuide-SDXL: Dual Guidance for Stable Diffusion XL (TPG + NAG)"""

__version__ = "1.0.0"

from .pipelines.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline
from .processors.custom_attn import (
    NAGCrossAttnProcessor,
    SoftPAGCrossAttnProcessor,
    HeadHunter,
    apply_nag,
    apply_softpag,
)

__all__ = [
    "StableDiffusionXLTPGPipeline",
    "NAGCrossAttnProcessor",
    "SoftPAGCrossAttnProcessor",
    "HeadHunter",
    "apply_nag",
    "apply_softpag",
]
