import math
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
        half_hidden_size: int = self.hidden_size // 2
        pow: Tensor = torch.arange(start=0, end=half_hidden_size, step=1, dtype=torch.float32) * -math.log(x=10000, base=2) / (half_hidden_size - 1)
        emb: Tensor = torch.exp(input=pow)
        h: Tensor = scale * h[..., None] *  emb[None, ...]
        h: Tensor = torch.concat([h, h], dim=-1)
        return h


def test_all() -> None:
    test_pre_lookahead_layer()

if __name__ == '__main__':
    test_all()
