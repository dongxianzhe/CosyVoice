import torch
from torch import Tensor

def add_optional_chunk_mask(masks: torch.Tensor) -> Tensor:
    # masks (B, 1, L)
    chunk_masks = masks
    assert chunk_masks.dtype == torch.bool
    if (chunk_masks.sum(dim=-1) == 0).sum().item() != 0:
        breakpoint()
        print('get chunk_masks all false at some timestep, force set to true, make sure they are masked in futuer computation!')
        chunk_masks[chunk_masks.sum(dim=-1) == 0] = True
    return chunk_masks


def make_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask
