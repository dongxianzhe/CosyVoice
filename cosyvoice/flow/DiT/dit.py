"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
from __future__ import annotations
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import repeat
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb
from cosyvoice.utils.mask import add_optional_chunk_mask
import math


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor, scale: float=1000) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class CausalConvPositionEmbedding(nn.Module):
    def __init__(self, dim: int, kernel_size: int=31, groups: int=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.kernel_size: int = kernel_size
        self.conv1= nn.Sequential(
            nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # x (b, n, d) mask (b, n) d is dim, n is time, b is batch
        if mask is not None:
            mask: Tensor = mask[..., None]
            x = x.masked_fill(~mask, 0.0)
        x = x.permute(0, 2, 1) # (b, d, n)
        x = nn.functional.pad(input=x, pad=(self.kernel_size - 1, 0, 0, 0)) # (b, d, n + kernel_size - 1) pad in left
        x = self.conv1(x) # (b, d, n)
        x = nn.functional.pad(input=x, pad=(self.kernel_size - 1, 0, 0, 0)) # (b, d, n + kernel_size - 1) pad in left
        x = self.conv2(x) # (b, d, n)
        out: Tensor = x.permute(0, 2, 1) # (b, n, d)
        if mask is not None:
            out = out.masked_fill(~mask, 0.0)
        return out # (b, n, d)


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(in_features=dim, out_features=dim * 6)
        self.norm = nn.LayerNorm(normalized_shape=dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: Tensor, emb=None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # x (b, n, dim) emb (b, dim)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(input=emb, chunks=6, dim=1) # (b, 6 * dim) -> (b, dim), (b, dim), (b, dim), (b, dim), (b, dim), (b, dim)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(in_features=dim, out_features=dim * 2)
        self.norm = nn.LayerNorm(normalized_shape=dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        # x (b, n, dim) emb (b, dim)
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(input=emb, chunks=2, dim=1)
        # scale (b, dim) shift (b, dim)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :] # (b, n, dim)
        return x # (b, n, dim)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int=4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = nn.Sequential(
            nn.Sequential(
                nn.Linear(dim, inner_dim), 
                nn.GELU(approximate="tanh")
            ), 
            nn.Dropout(0.1), 
            nn.Linear(inner_dim, dim), 
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ff(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads

        self.to_q = nn.Linear(in_features=dim, out_features=self.inner_dim)
        self.to_k = nn.Linear(in_features=dim, out_features=self.inner_dim)
        self.to_v = nn.Linear(in_features=dim, out_features=self.inner_dim)

        self.to_out = nn.ModuleList()
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(0.1))

    def forward(self, x: Tensor, mask: Tensor, rope: Tensor) -> torch.Tensor:
        # x (b, n, d) mask (b, n) rope (1, n, d)
        batch_size = x.shape[0]
        query, key, value = self.to_q(x), self.to_k(x), self.to_v(x)
        freqs, scale = rope
        query, key = apply_rotary_pos_emb(query, freqs), apply_rotary_pos_emb(key, freqs)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (b, n, d) -> (b, h, n, d)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (b, n, d) -> (b, h, n, d)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (b, n, d) -> (b, h, n, d)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = mask
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
                attn_mask = attn_mask.expand(batch_size, self.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim) # (b, h, n, d) -> (b, n, h * d)
        x = x.to(query.dtype)
        # linear proj
        x = self.to_out[0](x)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(dim=-1)
            else:
                mask = mask[:, 0, -1].unsqueeze(dim=-1)
            x = x.masked_fill(~mask, 0.0)
        return x


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: int):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult)

    def forward(self, x: Tensor, t: Tensor, mask: Tensor, rope: tuple[Tensor, float]) -> Tensor:  # x: noised input, t: time embedding
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output
        ff_norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(ff_norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output
        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: Tensor):  # noqa: F821
        # timestep (batch_size, )
        time_hidden = self.time_embed(timestep) # (batch_size, freq_embed_dim)
        time_hidden = time_hidden.to(timestep.dtype)
        return self.time_mlp(time_hidden)  # (batch_size, dim)


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, text_dim: int, out_dim: int, spk_dim: int):
        super().__init__()
        self.spk_dim = spk_dim
        self.proj = nn.Linear(mel_dim * 2 + text_dim + spk_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(dim=out_dim)

    def forward(self, x: Tensor, cond: Tensor, text_embed: Tensor, spks: Tensor) -> Tensor:
        # x (b, n, d) cond (b, n, d=80) text_embed (b, n, d=80) spks (b, d=80)
        to_cat: list[Tensor] = [x, cond, text_embed]
        if self.spk_dim > 0:
            spks = repeat(spks, "b c -> b t c", t=x.shape[1])
            to_cat.append(spks)
        x = self.proj(torch.cat(to_cat, dim=-1)) # (b, n, out_dim)
        return self.conv_pos_embed(x) + x # (b, n, out_dim)


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        ff_mult: int,
        mel_dim: int,
        mu_dim: int,
        spk_dim: int,
        out_channels: int,
        static_chunk_size: int,
        num_decoding_left_chunks: int, 
    ):
        super().__init__()
        self.time_embed = TimestepEmbedding(dim)
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.transformer_blocks = nn.ModuleList([DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult) for _ in range(depth)])
        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)
        self.static_chunk_size = static_chunk_size

    def forward(self, x: Tensor, mask: Tensor, mu: Tensor, t: Tensor, spks: Tensor, cond: Tensor, streaming: bool=False) -> Tensor: 
        # x (batch_size, hidden_size=80, mel_timesteps)
        # mask (batch_size, 1, mel_timesteps)
        # mu (batch_size, 1, mel_timesteps)
        # t (batch_size, )
        # spks shape: (batch_size, hidden_size=80)
        # cond (batch_size, hidden_size, mel_timesteps)
        x = x.transpose(1, 2) # (batch_size, mel_timesteps, hidden_size=80)
        mu = mu.transpose(1, 2) # (batch_size, mel_timesteps, 1)
        cond = cond.transpose(1, 2) # (batch_size, mel_timesteps, hidden_size=80)
        spks = spks.unsqueeze(dim=1) # (batch_size, 1, hidden_size=80)
        batch, seq_len = x.shape[0], x.shape[1] # batch_size, mel_timesteps
        if t.ndim == 0:
            t = t.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(t) # (batch_size, hidden_size=1024)
        x = self.input_embed(x, cond, mu, spks.squeeze(1)) # (batch_size, mel_timesteps, hidden_size=1024)

        rope: tuple[Tensor, float] = self.rotary_embed.forward_from_seq_len(seq_len) # (1, mel_timesteps, 64)

        if streaming is True:
            attn_mask: Tensor = add_optional_chunk_mask(x, mask.bool(), False, False, 0, self.static_chunk_size, -1).unsqueeze(dim=1)
        else:
            attn_mask: Tensor = add_optional_chunk_mask(x, mask.bool(), False, False, 0, 0, -1).repeat(1, x.size(1), 1).unsqueeze(dim=1) # (batch_size, 1, mel_timesteps, mel_timesteps)

        for block in self.transformer_blocks:
            x = block(x, t, mask=attn_mask.bool(), rope=rope) # (batch_size, mel_timesteps, hidden_size=1024)

        x = self.norm_out(x, t) # (batch_size, mel_timesteps, hidden_size=1024)
        return self.proj_out(x).transpose(1, 2) # (batch_size, hidden_size=80, mel_timesteps)
