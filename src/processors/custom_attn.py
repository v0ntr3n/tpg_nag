# ---- ENHANCED custom cross-attention (QK-Norm + text-gated), latest version-robust ----
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import (
    AttnProcessor2_0 as SDPProc,
    Attention
)
from typing import Optional, Dict, Any

# Prefer Flash/MemEff kernels if available
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
except Exception:
    pass


class NAGCrossAttnProcessor(nn.Module):
    """
    Normalized Attention Guidance (NAG) cross-attention processor.

    This implements a lightweight version of the method from
    "Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models"
    (arXiv:2505.21179). It works by computing attention maps for grouped batches
    (e.g., unconditional + conditional branches used by CFG), applying an L1
    normalization in attention space, and then extrapolating (refining) the
    conditional attention using the difference between conditional and
    unconditional attention maps.

    Notes:
    - This is an inference-time, training-free processor. It's a simplified
      version intended for easy integration and experimentation.
    - It targets cross-attention sites by design. For self-attention you can
      keep the original processors.
    """

    def __init__(self, alpha: float = 1.0, use_l1: bool = True, eps: float = 1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.use_l1 = bool(use_l1)
        self.eps = float(eps)

    def _num_heads(self, attn: Attention) -> int:
        if hasattr(attn, "heads"):
            return attn.heads
        if hasattr(attn, "num_heads"):
            return attn.num_heads
        # fallback
        return attn.to_q.out_features // getattr(attn, "head_dim", 64)

    def _head_dim(self, attn: Attention) -> int:
        if hasattr(attn, "head_dim"):
            return attn.head_dim
        if hasattr(attn, "dim_head"):
            return attn.dim_head
        n_heads = self._num_heads(attn)
        return attn.to_q.out_features // n_heads

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        """Processor callable compatible with Diffusers' Attention processors."""
        # Basic sanity: if no encoder states are provided treat as identity
        if encoder_hidden_states is None:
            # fall back to standard behaviour: use value projection of hidden_states
            return attn.to_out[0](attn.to_v(hidden_states))

        # Prepare shapes
        b, q_len, _ = hidden_states.shape
        # We assume encoder_hidden_states correspond to the base (non-duplicated)
        # batch; in CFG-style pipelines, hidden_states may contain groups where
        # batch = G * B_enc, where G is grouping factor (1,2,3)
        enc_b = encoder_hidden_states.shape[0]
        group = max(1, b // enc_b)

        # compute queries for the (possibly grouped) hidden_states
        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)

        n_heads = self._num_heads(attn)
        d_head = self._head_dim(attn)

        def pack(x, B):
            # x: (B_total, L, C) -> (group, B_enc, heads, L, d_head)
            x = x.view(group, B, -1, n_heads, d_head) if group > 1 else x.view(B, -1, n_heads, d_head)
            if group > 1:
                # make (G, B, H, L, Dh)
                x = x.permute(0, 1, 3, 2, 4)
            else:
                x = x.permute(0, 2, 1, 3)
            return x

        # Pack q (grouped by G), k and v (k/v are size enc_b)
        if group > 1:
            q_p = pack(q, b // group)
            k_p = pack(k.repeat(group, 1, 1) if k.shape[0] == enc_b else k, enc_b)
            v_p = pack(v.repeat(group, 1, 1) if v.shape[0] == enc_b else v, enc_b)
            # q_p: (G, B, H, Lq, Dh)
        else:
            # Single-group: use normal shapes
            q_p = q.view(b, -1, n_heads, d_head).permute(0, 2, 1, 3)  # (B, H, Lq, Dh)
            k_p = k.view(enc_b, -1, n_heads, d_head).permute(0, 2, 1, 3)  # (B_enc, H, Lk, Dh)
            v_p = v.view(enc_b, -1, n_heads, d_head).permute(0, 2, 1, 3)

        # We'll compute attention per group/branch and then apply normalization+extrapolation
        if group > 1:
            # iterate over group dimension
            outputs = []
            for g in range(group):
                qg = q_p[g]  # (B, H, Lq, Dh)
                kg = k_p[g]
                vg = v_p[g]
                B_enc = qg.shape[0]

                # Compute raw scores for each member in the group
                # reshape to (B*H, Lq, Dh) and (B*H, Lk, Dh)
                q_flat = qg.reshape(B_enc * n_heads, qg.shape[2], d_head)
                k_flat = kg.reshape(B_enc * n_heads, kg.shape[2], d_head)
                scores = torch.bmm(q_flat, k_flat.transpose(1, 2)) / math.sqrt(d_head)
                probs = torch.softmax(scores, dim=-1)

                # reshape back (B, H, Lq, Lk)
                probs = probs.view(B_enc, n_heads, qg.shape[2], kg.shape[2])

                # If group >=2 we expect at least uncond (index 0) and cond (index 1)
                if B_enc >= 2:
                    base = probs[0]
                    refined_list = []
                    for idx in range(B_enc):
                        cur = probs[idx]
                        if self.use_l1:
                            # L1 normalization across key dim (keep sum to 1 but emphasize sparsity)
                            cur_norm = cur / (cur.abs().sum(dim=-1, keepdim=True) + self.eps)
                            base_norm = base / (base.abs().sum(dim=-1, keepdim=True) + self.eps)
                        else:
                            cur_norm = cur / (cur.sum(dim=-1, keepdim=True) + self.eps)
                            base_norm = base / (base.sum(dim=-1, keepdim=True) + self.eps)

                        if idx == 0:
                            # unconditional unchanged
                            refined = cur
                        else:
                            # extrapolate in attention space
                            refined = cur_norm + self.alpha * (cur_norm - base_norm)
                            # renormalize to be a proper attention map
                            refined = torch.softmax(refined.view(-1, refined.shape[-1]), dim=-1).view(refined.shape)

                        refined_list.append(refined)

                    # compute outputs for each branch
                    out_branches = []
                    for idx in range(B_enc):
                        refined = refined_list[idx]
                        # refined: (H, Lq, Lk) -> put heads into batch to multiply with v
                        # compute attention output per head using refined attention maps
                        out_head = torch.matmul(refined, vg[idx])  # (H, Lq, Dh)
                        out_head = out_head.permute(1, 0, 2).reshape(qg.shape[0], qg.shape[2], n_heads * d_head)
                        out_branches.append(out_head)

                    # stack branches back into (B, Lq, C)
                    out = torch.stack(out_branches, dim=0)
                    outputs.append(out)
                else:
                    # single-branch inside group (degenerate), just compute standard attention output
                    out = F.scaled_dot_product_attention(
                        qg.reshape(B_enc * n_heads, qg.shape[2], d_head),
                        kg.reshape(B_enc * n_heads, kg.shape[2], d_head),
                        vg.reshape(B_enc * n_heads, vg.shape[2], d_head),
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False,
                    )
                    out = out.view(B_enc, n_heads, qg.shape[2], d_head).permute(0, 2, 1, 3).reshape(B_enc, qg.shape[2], n_heads * d_head)
                    outputs.append(out)

            # concatenate along batch (group dim collapsed)
            final = torch.cat(outputs, dim=0)

        else:
            # no grouping: standard attention forward using scaled_dot_product_attention
            out = F.scaled_dot_product_attention(
                q_p.reshape(b * n_heads, q_p.shape[2], d_head),
                k_p.reshape(enc_b * n_heads, k_p.shape[2], d_head),
                v_p.reshape(enc_b * n_heads, v_p.shape[2], d_head),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            final = out.view(b, n_heads, q_p.shape[2], d_head).permute(0, 2, 1, 3).reshape(b, q_p.shape[2], n_heads * d_head)

        # final projection
        final = attn.to_out[0](final)
        final = attn.to_out[1](final)
        return final


def apply_nag(pipe, alpha: float = 1.0, use_l1: bool = True):
    """
    Replace cross-attention processors in `pipe.unet` with NAG processors.

    Returns the pipeline for convenience. This will only modify processors whose
    names contain `attn2` (convention used in Diffusers for cross-attention).
    """
    try:
        if not hasattr(pipe, "unet"):
            raise AttributeError("Pipeline has no attribute 'unet'")

        current = getattr(pipe.unet, "attn_processors", {})
        procs = {}
        applied = 0
        for name, proc in current.items():
            if "attn2" in name or name.endswith("cross_attn") or "cross" in name:
                procs[name] = NAGCrossAttnProcessor(alpha=alpha, use_l1=use_l1)
                applied += 1
            else:
                procs[name] = proc

        if applied == 0:
            print("[NAG] No cross-attention processors detected to replace. No changes applied.")
            return pipe

        pipe.unet.set_attn_processor(procs)
        print(f"[NAG] Applied NAGCrossAttnProcessor to {applied} sites (alpha={alpha}, use_l1={use_l1}).")
        return pipe
    except Exception as e:
        print(f"[NAG] Error applying NAG processors: {e}")
        return pipe


class SoftPAGCrossAttnProcessor(nn.Module):
    """SoftPAG: softly interpolate selected attention heads towards identity.

    This processor modifies attention maps for selected heads as:
        A_h' = (1 - beta) * A_h + beta * I
    where I is an identity-like matrix (query i attends to key i). If
    `selected_heads` is None, the interpolation is applied to all heads.
    """

    def __init__(self, beta: float = 0.5, selected_heads: Optional[list] = None):
        super().__init__()
        self.beta = float(beta)
        self.selected_heads = None if selected_heads is None else list(selected_heads)

    def _num_heads(self, attn: Attention) -> int:
        if hasattr(attn, "heads"):
            return attn.heads
        if hasattr(attn, "num_heads"):
            return attn.num_heads
        return attn.to_q.out_features // getattr(attn, "head_dim", 64)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        # Fallback to identity on missing encoder states
        if encoder_hidden_states is None:
            return attn.to_out[0](attn.to_v(hidden_states))

        b, q_len, _ = hidden_states.shape
        enc_b = encoder_hidden_states.shape[0]

        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)

        n_heads = self._num_heads(attn)
        # assume head_dim inferred from k
        d_head = k.shape[-1] // n_heads

        # reshape: (B, L, C) -> (B, H, L, Dh)
        q = q.view(b, -1, n_heads, d_head).permute(0, 2, 1, 3)
        k = k.view(enc_b, -1, n_heads, d_head).permute(0, 2, 1, 3)
        v = v.view(enc_b, -1, n_heads, d_head).permute(0, 2, 1, 3)

        # compute scores and probs per head
        # We'll handle grouped batches simply by broadcasting k/v if necessary
        if enc_b == b:
            query = q.reshape(b * n_heads, q.shape[2], d_head)
            key = k.reshape(b * n_heads, k.shape[2], d_head)

            scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d_head)
            probs = torch.softmax(scores, dim=-1)
            probs = probs.view(b, n_heads, q.shape[2], k.shape[2])

            # build identity matrix of shape (Lq, Lk)
            Lq, Lk = q.shape[2], k.shape[2]
            eye = torch.eye(Lk, device=probs.device, dtype=probs.dtype)
            if Lq != Lk:
                # crop or pad eye to (Lq, Lk) by taking first Lq rows
                eye = eye[:Lq, :]

            apply_all = self.selected_heads is None
            for h in range(n_heads):
                if apply_all or (h in self.selected_heads):
                    probs[:, h, :, :] = (1.0 - self.beta) * probs[:, h, :, :] + self.beta * eye.unsqueeze(0)

            # compute outputs
            out = torch.matmul(probs, v)  # (B, H, Lq, Dh)
            out = out.permute(0, 2, 1, 3).reshape(b, q.shape[2], n_heads * d_head)
        else:
            # broadcast encoder batch to match query batch (common when using CFG stacking)
            # simple approach: repeat k and v to match b
            k_rep = k.repeat(int(b / enc_b), 1, 1, 1) if b % enc_b == 0 else k
            v_rep = v.repeat(int(b / enc_b), 1, 1, 1) if b % enc_b == 0 else v

            query = q.reshape(b * n_heads, q.shape[2], d_head)
            key = k_rep.reshape(b * n_heads, k_rep.shape[2], d_head)

            scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d_head)
            probs = torch.softmax(scores, dim=-1).view(b, n_heads, q.shape[2], k_rep.shape[2])

            Lq, Lk = q.shape[2], k_rep.shape[2]
            eye = torch.eye(Lk, device=probs.device, dtype=probs.dtype)
            if Lq != Lk:
                eye = eye[:Lq, :]

            apply_all = self.selected_heads is None
            for h in range(n_heads):
                if apply_all or (h in self.selected_heads):
                    probs[:, h, :, :] = (1.0 - self.beta) * probs[:, h, :, :] + self.beta * eye.unsqueeze(0)

            out = torch.matmul(probs, v_rep)
            out = out.permute(0, 2, 1, 3).reshape(b, q.shape[2], n_heads * d_head)

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


class HeadHunter:
    """Utility to compute simple per-head scores from attention maps.

    This class provides a static method `score_heads(attn, hidden_states, encoder_hidden_states)`
    that computes a per-head score (higher = more "important") using the variance
    of attention maps across tokens. It's a heuristic used to select heads for
    SoftPAG. For a production-grade selection run an actual validation set and
    compute task-correlated metrics.
    """

    @staticmethod
    def score_heads(attn: Attention, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # projects
        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states)
        n_heads = getattr(attn, 'heads', getattr(attn, 'num_heads', None))
        if n_heads is None:
            n_heads = q.shape[-1] // getattr(attn, 'head_dim', 64)
        d_head = q.shape[-1] // n_heads

        q = q.view(q.shape[0], -1, n_heads, d_head).permute(0, 2, 1, 3)  # (B, H, Lq, Dh)
        k = k.view(k.shape[0], -1, n_heads, d_head).permute(0, 2, 1, 3)  # (B, H, Lk, Dh)

        # compute scores per head
        B, H, Lq, Dh = q.shape
        _, _, Lk, _ = k.shape
        q_flat = q.reshape(B * H, Lq, Dh)
        k_flat = k.reshape(B * H, Lk, Dh)
        scores = torch.bmm(q_flat, k_flat.transpose(1, 2)) / math.sqrt(Dh)
        probs = torch.softmax(scores, dim=-1)
        probs = probs.view(B, H, Lq, Lk)

        # score heads by mean attention entropy or variance across keys
        # here use variance across keys averaged over queries and batch
        var = probs.var(dim=-1).mean(dim=-1).mean(dim=0)  # (H,)
        return var


def apply_softpag(pipe, beta: float = 0.5, selected_heads: Optional[list] = None):
    """Apply SoftPAG processors to cross-attention sites in the UNet.

    Arguments:
        pipe: pipeline with `unet.attn_processors`
        beta: interpolation factor toward identity (0=no change, 1=identity)
        selected_heads: list of head indices to apply; None => apply to all heads
    """
    try:
        if not hasattr(pipe, 'unet'):
            raise AttributeError('Pipeline has no attribute unet')
        current = getattr(pipe.unet, 'attn_processors', {})
        procs = {}
        applied = 0
        for name, proc in current.items():
            if 'attn2' in name or name.endswith('cross_attn') or 'cross' in name:
                procs[name] = SoftPAGCrossAttnProcessor(beta=beta, selected_heads=selected_heads)
                applied += 1
            else:
                procs[name] = proc
        if applied == 0:
            print('[SoftPAG] No cross-attention processors matched. No changes applied.')
            return pipe
        pipe.unet.set_attn_processor(procs)
        print(f"[SoftPAG] Applied to {applied} cross-attention sites (beta={beta}).")
        return pipe
    except Exception as e:
        print(f"[SoftPAG] Error applying processors: {e}")
        return pipe

# Usage example:
# pipe = apply_enhanced_custom_attention(pipe, qk_norm=True, gate_heads=True, scale=1.0, use_rms_norm=False)