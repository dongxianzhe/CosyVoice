import math
from torch._tensor import Tensor
from torch.nn.modules.linear import Linear
from torch._tensor import Tensor
from torch._tensor import Tensor
from torch._tensor import Tensor
from torch.nn.modules.container import Sequential
from torch.nn.modules.container import Sequential
from torch.nn.modules.container import Sequential
import torch
from torch import nn, Tensor
from dataclasses import dataclass

@dataclass
class PreLookaheadLayerConfig:
    in_channels: int
    intermediate_channels: int
    pre_lookahead_len: int = 1

class PreLookaheadLayer(nn.Module):
    def __init__(self, config: PreLookaheadLayerConfig) -> None:
        super().__init__()
        self.config: PreLookaheadLayerConfig = config
        self.conv1: nn.Conv1d = nn.Conv1d(
            in_channels=self.config.in_channels,
            out_channels=self.config.intermediate_channels,
            kernel_size=self.config.pre_lookahead_len + 1,
            stride=1,
            padding=0,
        )
        self.conv2: nn.Conv1d = nn.Conv1d(
            in_channels=self.config.intermediate_channels,
            out_channels=self.config.in_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )
    
    def forward(self, h: Tensor) -> Tensor:
        """
        h: (batch_size, seq_len, channels)
        """
        h: Tensor = h.transpose(1, 2).contiguous() # (batch_size, channels, seq_len)
        h: Tensor = nn.functional.pad(input=h, pad=(0, self.config.pre_lookahead_len), mode='constant', value=0.0) # (batch_size, channels, seq_len + pre_lookahead_len)
        h: Tensor = self.conv1(h) # (batch_size, intermediate_channels, seq_len)
        h: Tensor = nn.functional.pad(input=h, pad=(self.conv2.kernel_size[0] - 1, 0), mode='constant', value=0.0) # (batch_size, intermediate_channels, seq_len + 2)
        h: Tensor = self.conv2(h) # (batch_size, in_channels, seq_len)
        h: Tensor = h.transpose(1, 2).contiguous() # (batch_size, seq_len, in_channels)
        return h


def test_pre_lookahead_layer() -> None:
    config: PreLookaheadLayerConfig = PreLookaheadLayerConfig(
        in_channels=512,
        intermediate_channels=1024,
        pre_lookahead_len=1,
    )
    layer: PreLookaheadLayer = PreLookaheadLayer(config=config)
    h: Tensor = torch.randn(1, 100, 512)
    h: Tensor = layer(h)
    assert h.shape == (1, 100, 512)


class SinusPositionEmbedding(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
    
    def forward(self, h: Tensor, scale: float=1000.0) -> Tensor:
        # 假设 hidden_size=256, h 输入形状为 (B,), 例如 B=2
        # h: (B,)  例如 (2,)

        half_hidden_size: int = self.hidden_size // 2  # 128

        pow: Tensor = torch.arange(start=0, end=half_hidden_size, step=1, dtype=torch.float32) * -math.log(x=10000, base=2) / (half_hidden_size - 1)
        # torch.arange(0, 128): (128,)
        # * -log(10000) / 127:  (128,)  — 每个位置的频率指数, 从 0 递减到 -log(10000)
        # pow: (128,)

        emb: Tensor = torch.exp(input=pow)
        # exp(pow): (128,)  — 频率系数, 从 1.0 指数衰减到 1/10000
        # emb: (128,)

        h: Tensor = scale * h[..., None] *  emb[None, ...]
        # h[..., None]:  (B,) -> (B, 1)        — 扩展最后一维
        # emb[None, ...]: (128,) -> (1, 128)   — 扩展第一维
        # 广播相乘: (B, 1) * (1, 128) -> (B, 128)
        # * scale: (B, 128)
        # h: (B, 128)

        h: Tensor = torch.concat([h, h], dim=-1)
        # [h, h] 沿最后一维拼接: (B, 128) cat (B, 128) -> (B, 256)
        # h: (B, 256) 即 (B, hidden_size)

        return h  # (B, hidden_size)


class TimestepEmbedding(nn.Module):
    def __init__(self, output_dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed: SinusPositionEmbedding = SinusPositionEmbedding(hidden_size=freq_embed_dim)
        self.time_mlp: Sequential = nn.Sequential(
            nn.Linear(in_features=freq_embed_dim, out_features=output_dim), 
            nn.SiLU(), 
            nn.Linear(in_features=output_dim, out_features=output_dim)
        )

    def forward(self, timestep: Tensor) -> Tensor:
        # timestep: (B,)
        time_hidden: Tensor = self.time_embed(timestep).to(timestep.dtype) # (B, freq_embed_dim)
        time = self.time_mlp(time_hidden) # (B, hidden_size)
        return time


class CausalConvPositionEmbedding(nn.Module):
    def __init__(self, dim: int, kernel_size: int=31, groups: int=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.kernel_size: int = kernel_size
        self.conv1: nn.Sequential = nn.Sequential(
            nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )
        self.conv2: Sequential = nn.Sequential(
            nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # x (b, n, d) mask (b, n) d is dim, n is time, b is batch
        if mask is not None:
            mask: Tensor = mask[..., None]
            x: Tensor = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1) # (b, d, n)
        x = nn.functional.pad(input=x, pad=(self.kernel_size - 1, 0, 0, 0)) # (b, d, n + kernel_size - 1) pad in left
        x = self.conv1(x) # (b, d, n)
        x = nn.functional.pad(input=x, pad=(self.kernel_size - 1, 0, 0, 0)) # (b, d, n + kernel_size - 1) pad in left
        x = self.conv2(x) # (b, d, n)
        out: Tensor = x.permute(0, 2, 1) # (b, n, d)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out # (b, n, d)


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, text_dim: int, out_dim: int, spk_dim: int=0):
        super().__init__()
        self.spk_dim: int = spk_dim
        self.proj: nn.Linear = nn.Linear(in_features=mel_dim * 2 + text_dim + spk_dim, out_features=out_dim)
        self.conv_pos_embed: CausalConvPositionEmbedding = CausalConvPositionEmbedding(dim=out_dim)

    def forward(self, x: Tensor, cond: Tensor, text_embed: Tensor, spks: Tensor, ):
        # x (b, n, mel_dim) cond (b, n, mel_dim) text_embed (b, n, text_dim) spks(b, d) 
        to_cat: list[Tensor] = [x, cond, text_embed]
        if self.spk_dim > 0:
            spks: Tensor = spks[:, None, :].expand(-1, x.shape[1], -1)  # (b, c) -> (b, t, c)
            to_cat.append(spks)

        x = self.proj(torch.cat(to_cat, dim=-1)) # (b, n, out_dim)
        x = self.conv_pos_embed(x) + x # (b, n, out_dim)
        return x # (b, n, out_dim)


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


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int, mult: int=4, approximate: str = "none"):
        super().__init__()
        inner_dim: int = int(dim * mult)
        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(in_features=dim, out_features=inner_dim), activation)
        self.ff = nn.Sequential(
            project_in, 
            nn.Linear(in_features=inner_dim, out_features=dim_out)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x (b, n, dim)
        return self.ff(x) # (b, n, dim_out)


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

def test_all() -> None:
    test_pre_lookahead_layer()


if __name__ == '__main__':
    test_all()
