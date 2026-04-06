import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.nn.utils.parametrizations import weight_norm, spectral_norm
except ImportError:
    from torch.nn.utils import weight_norm, spectral_norm
from typing import List, Optional, Tuple
from einops import rearrange
from torchaudio.transforms import Spectrogram

LRELU_SLOPE = 0.1


class MultipleDiscriminator(nn.Module):
    def __init__(
            self, mpd: nn.Module, mrd: nn.Module
    ):
        super().__init__()
        self.mpd = mpd
        self.mrd = mrd


class MultiResSpecDiscriminator(torch.nn.Module):
    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window"):

        super(MultiResSpecDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            SpecDiscriminator(fft_sizes[0], hop_sizes[0], win_lengths[0], window),
            SpecDiscriminator(fft_sizes[1], hop_sizes[1], win_lengths[1], window),
            SpecDiscriminator(fft_sizes[2], hop_sizes[2], win_lengths[2], window)])


class SpecDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window", use_spectral_norm=False):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.discriminators = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ])

        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))