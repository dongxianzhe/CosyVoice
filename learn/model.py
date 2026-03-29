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



if __name__ == '__main__':
    pass