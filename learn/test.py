import torch
from typing import Any
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'Matcha-TTS')))
from cosyvoice.tokenizer.tokenizer import CosyVoice2Tokenizer, CosyVoice3Tokenizer, get_qwen_tokenizer
from matcha.utils.audio import mel_spectrogram
from functools import partial

# ============================== tokenizer ==============================
def test_tokenizer() -> None:
    tokenizer: CosyVoice2Tokenizer | CosyVoice3Tokenizer = get_qwen_tokenizer(
        token_path="/data/home/xianzhedong/models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN", 
        skip_special_tokens=True,
        version='cosyvoice3', 
    )

    tokens: list[int] = tokenizer.encode(text='今天天气真好呀，你能不能陪我出去逛逛？')
    tokens_ref: list[int] = [36171,  35727,  35727,  99180,  88051,  52801, 104256,   3837,  56568, 26232,  16530,  26232, 100522,  35946,  20221,  85336, 102946, 102946, 11319]
    assert tokens == tokens_ref

    print(f'text tokens {len(tokens)} {tokens}')

    tokens: Any | list[int] = tokenizer.encode(text='You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。')
    tokens_ref: list[int] = [2610,    525,    264,  10950,  17847,     13, 151646,  99658,  99317, 56568,  23031,  33447,  26232,  99521,  99190,   9370,  56006,  35946, 97706,  52801, 119024,   1773]
    assert tokens == tokens_ref
    print(f'prompt text tokens {len(tokens)} {tokens}')

# ============================== feat extractor ==============================
def test_feat_extractor() -> None:
    n_fft=1920
    num_mels=80
    sampling_rate=24000
    hop_size=480
    win_size=1920
    fmin=0
    fmax=None
    center=False
    sample_rate = 24000
    samples = 83520
    n_channel = 1
    feat_extractor: Any = partial(mel_spectrogram, n_fft=n_fft, num_mels=num_mels, sampling_rate=sampling_rate, hop_size=hop_size, win_size=win_size, fmin=fmin, fmax=fmax, center=center)
    speech = torch.randn(size=(n_channel, samples), dtype=torch.float)
    speech_feat = feat_extractor(speech)
    print(f'speech_feat.shape {speech_feat.shape}')
    assert speech_feat.shape[0] == n_channel
    assert speech_feat.shape[1] == num_mels
    assert speech_feat.shape[2] == (samples + n_fft - hop_size - win_size) // hop_size + 1

# test_tokenizer()
test_feat_extractor()