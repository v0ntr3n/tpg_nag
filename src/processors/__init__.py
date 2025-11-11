"""Custom attention processors for TPG-NAG"""

from .custom_attn import (
    NAGCrossAttnProcessor,
    SoftPAGCrossAttnProcessor,
    HeadHunter,
    apply_nag,
    apply_softpag,
)

__all__ = [
    "NAGCrossAttnProcessor",
    "SoftPAGCrossAttnProcessor",
    "HeadHunter",
    "apply_nag",
    "apply_softpag",
]
