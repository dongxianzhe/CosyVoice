from __future__ import annotations
import torchaudio
import time
from transformers import AutoTokenizer
from wetext import Normalizer
import math
import os
import re
import regex
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Callable, Any, Generator
import numpy as np
import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
import torchaudio.compliance.kaldi as kaldi
from einops import repeat
from scipy.signal import get_window
from torch import Tensor, nn, sin, pow
from tqdm import tqdm
from transformers import Qwen2ForCausalLM
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm


def _rotate_half(x: Tensor) -> Tensor:
    """将向量的前后两半交换并取负，用于 RoPE"""
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(t: Tensor, freqs: Tensor) -> Tensor:
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype
    freqs = freqs[:, -seq_len:]
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = freqs.unsqueeze(1)  # (b, 1, n, d)
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t_rot = t_rot * freqs.cos() + _rotate_half(t_rot) * freqs.sin()
    return torch.cat((t_rot, t_pass), dim=-1).to(orig_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward_from_seq_len(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t.float(), self.inv_freq)  # (seq_len, dim//2)
        freqs = torch.stack((freqs, freqs), dim=-1).flatten(-2)     # (seq_len, dim)
        return freqs.unsqueeze(0), 1.0  # (1, seq_len, dim), scale=1.0


class CosyVoice3Tokenizer:
    def __init__(self, token_path, skip_special_tokens=True):
        # NOTE: non-chat model, all these special tokens keep randomly initialized.
        special_tokens = {
            'eos_token': '<|endoftext|>',
            'pad_token': '<|endoftext|>',
            'additional_special_tokens': [
                '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]',
                '[laughter]', '[cough]', '[clucking]', '[accent]',
                '[quick_breath]',
                "<laughter>", "</laughter>",
                "[hissing]", "[sigh]", "[vocalized-noise]",
                "[lipsmack]", "[mn]", "<|endofsystem|>",
                "[AA]", "[AA0]", "[AA1]", "[AA2]", "[AE]", "[AE0]", "[AE1]", "[AE2]", "[AH]", "[AH0]", "[AH1]", "[AH2]",
                "[AO]", "[AO0]", "[AO1]", "[AO2]", "[AW]", "[AW0]", "[AW1]", "[AW2]", "[AY]", "[AY0]", "[AY1]", "[AY2]",
                "[B]", "[CH]", "[D]", "[DH]", "[EH]", "[EH0]", "[EH1]", "[EH2]", "[ER]", "[ER0]", "[ER1]", "[ER2]", "[EY]",
                "[EY0]", "[EY1]", "[EY2]", "[F]", "[G]", "[HH]", "[IH]", "[IH0]", "[IH1]", "[IH2]", "[IY]", "[IY0]", "[IY1]",
                "[IY2]", "[JH]", "[K]", "[L]", "[M]", "[N]", "[NG]", "[OW]", "[OW0]", "[OW1]", "[OW2]", "[OY]", "[OY0]",
                "[OY1]", "[OY2]", "[P]", "[R]", "[S]", "[SH]", "[T]", "[TH]", "[UH]", "[UH0]", "[UH1]", "[UH2]", "[UW]",
                "[UW0]", "[UW1]", "[UW2]", "[V]", "[W]", "[Y]", "[Z]", "[ZH]",
                "[a]", "[ai]", "[an]", "[ang]", "[ao]", "[b]", "[c]", "[ch]", "[d]", "[e]", "[ei]", "[en]", "[eng]", "[f]",
                "[g]", "[h]", "[i]", "[ian]", "[in]", "[ing]", "[iu]", "[ià]", "[iàn]", "[iàng]", "[iào]", "[iá]", "[ián]",
                "[iáng]", "[iáo]", "[iè]", "[ié]", "[iòng]", "[ióng]", "[iù]", "[iú]", "[iā]", "[iān]", "[iāng]", "[iāo]",
                "[iē]", "[iě]", "[iōng]", "[iū]", "[iǎ]", "[iǎn]", "[iǎng]", "[iǎo]", "[iǒng]", "[iǔ]", "[j]", "[k]", "[l]",
                "[m]", "[n]", "[o]", "[ong]", "[ou]", "[p]", "[q]", "[r]", "[s]", "[sh]", "[t]", "[u]", "[uang]", "[ue]",
                "[un]", "[uo]", "[uà]", "[uài]", "[uàn]", "[uàng]", "[uá]", "[uái]", "[uán]", "[uáng]", "[uè]", "[ué]", "[uì]",
                "[uí]", "[uò]", "[uó]", "[uā]", "[uāi]", "[uān]", "[uāng]", "[uē]", "[uě]", "[uī]", "[uō]", "[uǎ]", "[uǎi]",
                "[uǎn]", "[uǎng]", "[uǐ]", "[uǒ]", "[vè]", "[w]", "[x]", "[y]", "[z]", "[zh]", "[à]", "[ài]", "[àn]", "[àng]",
                "[ào]", "[á]", "[ái]", "[án]", "[áng]", "[áo]", "[è]", "[èi]", "[èn]", "[èng]", "[èr]", "[é]", "[éi]", "[én]",
                "[éng]", "[ér]", "[ì]", "[ìn]", "[ìng]", "[í]", "[ín]", "[íng]", "[ò]", "[òng]", "[òu]", "[ó]", "[óng]", "[óu]",
                "[ù]", "[ùn]", "[ú]", "[ún]", "[ā]", "[āi]", "[ān]", "[āng]", "[āo]", "[ē]", "[ēi]", "[ēn]", "[ēng]", "[ě]",
                "[ěi]", "[ěn]", "[ěng]", "[ěr]", "[ī]", "[īn]", "[īng]", "[ō]", "[ōng]", "[ōu]", "[ū]", "[ūn]", "[ǎ]", "[ǎi]",
                "[ǎn]", "[ǎng]", "[ǎo]", "[ǐ]", "[ǐn]", "[ǐng]", "[ǒ]", "[ǒng]", "[ǒu]", "[ǔ]", "[ǔn]", "[ǘ]", "[ǚ]", "[ǜ]"
            ]
        }
        self.special_tokens = special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        self.tokenizer.add_special_tokens(special_tokens)
        self.skip_special_tokens = skip_special_tokens

    def encode(self, text, **kwargs):
        tokens = self.tokenizer([text], return_tensors="pt")
        tokens = tokens["input_ids"][0].cpu().tolist()
        return tokens

    def decode(self, tokens): 
        tokens = torch.tensor(tokens, dtype=torch.int64) 
        text = self.tokenizer.batch_decode([tokens], skip_special_tokens=self.skip_special_tokens)[0] 
        return text


@lru_cache(maxsize=None)
def get_qwen_tokenizer(
    token_path: str,
    skip_special_tokens: bool,
    version: str = 'cosyvoice2'
):
    if version == 'cosyvoice3':
        return CosyVoice3Tokenizer(token_path=token_path, skip_special_tokens=skip_special_tokens)
    else:
        raise ValueError


chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')

# whether contain chinese character
def contains_chinese(text):
    return bool(chinese_char_pattern.search(text))


# replace special symbol
def replace_corner_mark(text):
    text = text.replace('²', '平方')
    text = text.replace('³', '立方')
    return text


# remove meaningless symbol
def remove_bracket(text):
    text = text.replace('（', '').replace('）', '')
    text = text.replace('【', '').replace('】', '')
    text = text.replace('`', '').replace('`', '')
    text = text.replace("——", " ")
    return text


# split paragrah logic：
# 1. per sentence max len token_max_n, min len token_min_n, merge if last sentence len less than merge_len
# 2. cal sentence len according to lang
# 3. split sentence according to puncatation
def split_paragraph(text: str, tokenize, lang="zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False) -> list[str]:
    def calc_utt_length(_text: str):
        if lang == "zh":
            return len(_text)
        else:
            return len(tokenize(_text))

    def should_merge(_text: str):
        if lang == "zh":
            return len(_text) < merge_len
        else:
            return len(tokenize(_text)) < merge_len

    if lang == "zh":
        pounc = ['。', '？', '！', '；', '：', '、', '.', '?', '!', ';']
    else:
        pounc = ['.', '?', '!', ';', ':']
    if comma_split:
        pounc.extend(['，', ','])

    if text[-1] not in pounc:
        if lang == "zh":
            text += "。"
        else:
            text += "."

    st = 0
    utts = []
    for i, c in enumerate(text):
        if c in pounc:
            if len(text[st: i]) > 0:
                utts.append(text[st: i] + c)
            if i + 1 < len(text) and text[i + 1] in ['"', '”']:
                tmp = utts.pop(-1)
                utts.append(tmp + text[i + 1])
                st = i + 2
            else:
                st = i + 1

    final_utts: list[str] = []
    cur_utt = ""
    for utt in utts:
        if calc_utt_length(cur_utt + utt) > token_max_n and calc_utt_length(cur_utt) > token_min_n:
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    if len(cur_utt) > 0:
        if should_merge(cur_utt) and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            final_utts.append(cur_utt)

    return final_utts


# remove blank between chinese character
def replace_blank(text: str) -> str:
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if ((text[i + 1].isascii() and text[i + 1] != " ") and
                    (text[i - 1].isascii() and text[i - 1] != " ")):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def is_only_punctuation(text: str):
    # Regular expression: Match strings that consist only of punctuation marks or are empty.
    punctuation_pattern = r'^[\p{P}\p{S}]*$'
    return bool(regex.fullmatch(punctuation_pattern, text))


def load_wav(wav: str, target_sr: int, min_sr: int=16000):
    speech, sample_rate = torchaudio.load(wav, backend='soundfile') # (channels, frames)
    speech = speech.mean(dim=0, keepdim=True) # (channels=1, frames)
    if sample_rate != target_sr:
        assert sample_rate >= min_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech) # (1, frames / sample_rate * target_sr)
    return speech

class Timer:
    def __init__(self, name: str = ""):
        self.name: str = name
        self.start: float = 0.
        self.cost: float = 0.

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.cost = time.time() - self.start
        print(f"[{self.name}] cost {self.cost:.4f}s")


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def _mel_filter_bank(sr, n_fft, n_mels, fmin, fmax):
    """生成 Mel 滤波器组矩阵，shape (n_mels, n_fft//2+1)，等价于 librosa.filters.mel"""
    fmax = fmax or sr / 2.0
    n_freqs = n_fft // 2 + 1
    # n_mels+2 个等间距的 mel 点，转回 Hz
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    # 对应的 FFT bin 索引（浮点）
    bins = hz_points / (sr / n_fft)
    fft_freqs = np.arange(n_freqs)
    weights = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lower, center, upper = bins[i], bins[i + 1], bins[i + 2]
        # 上升斜坡
        weights[i] = np.where(fft_freqs < center,
                              (fft_freqs - lower) / (center - lower + 1e-10),
                              (upper - fft_freqs) / (upper - center + 1e-10))
        weights[i] = np.maximum(weights[i], 0)
    # Slaney 归一化：每个滤波器除以其带宽（mel 间距对应的 Hz 宽度）
    enorm = 2.0 / (hz_points[2:n_mels+2] - hz_points[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights


_mel_basis = {}
_hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    global _mel_basis, _hann_window
    key = f"{fmax}_{y.device}"
    if key not in _mel_basis:
        mel = _mel_filter_bank(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
        _hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
    y = F.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect").squeeze(1)
    spec = torch.view_as_real(torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size,
        window=_hann_window[str(y.device)], center=center,
        pad_mode="reflect", normalized=False, onesided=True, return_complex=True,
    ))
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(_mel_basis[key], spec)
    return torch.log(torch.clamp(spec, min=1e-5))


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
    for i in range(len(sorted_idx)):
        # sampling both top-p and numbers.
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    top_ids = indices[prob.multinomial(1, replacement=True)].item()
    return top_ids


def random_sampling(weighted_scores, decoded_tokens, sampling):
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True).item()
    return top_ids


def ras_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
    if rep_num >= win_size * tau_r:
        weighted_scores[top_ids] = -float('inf')
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids


@dataclass
class TTSInputParams:
    text: Tensor
    text_len: Tensor
    flow_embedding: Tensor
    llm_embedding: Tensor
    prompt_text: Tensor
    prompt_text_len: Tensor
    llm_prompt_speech_token: Tensor
    llm_prompt_speech_token_len: Tensor
    flow_prompt_speech_token: Tensor
    flow_prompt_speech_token_len: Tensor
    prompt_speech_feat: Tensor
    prompt_speech_feat_len: Tensor
    source_speech_token: Tensor = torch.zeros(1, 0, dtype=torch.int32)
    stream: bool = False
    speed: float = 1.0


@dataclass
class FlowInputParams:
    token: Tensor
    token_len: Tensor
    prompt_token: Tensor
    prompt_token_len: Tensor
    prompt_feat: Tensor
    prompt_feat_len: Tensor
    embedding: Tensor
    streaming: bool
    finalize: bool


def print_params(data: dict[str, Any], name: str): 
    print(f'{name}: ')
    for field, value in data.__dict__.items():
        if isinstance(value, Tensor) and value.numel() <= 16:
            print(f"    {field}: shape = {value.shape} value = {value}")
        elif isinstance(value, Tensor):
            print(f"    {field}: shape = {value.shape}")
        else:
            print(f"    {field}: {value}")


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path: str) -> None:
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrain_path)

    def forward_one_step(self, xs: Tensor, masks: Tensor, cache=None):
        # first  xs (1, n, 896) masks (1, n, n) 
        # second xs (1, 1, 196)
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache


class CosyVoice3LM(torch.nn.Module):
    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling: Callable,
    ):
        torch.nn.Module.__init__(self)
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        self.sos: int = speech_token_size + 0
        self.eos_token: int = speech_token_size + 1
        self.task_id: int = speech_token_size + 2
        self.fill_token: int = speech_token_size + 3
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 200, bias=False)
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 200, llm_input_size)
        self.sampling = sampling
        self.stop_token_ids = [speech_token_size + i for i in range(200)]

    @torch.inference_mode()
    def inference(self, params: TTSInputParams, sampling: int = 25, max_token_text_ratio: float = 20, min_token_text_ratio: float = 2):
        text = torch.concat([params.prompt_text, params.text], dim=1)
        text_len = params.text_len + params.prompt_text_len
        text_emb = self.llm.model.model.embed_tokens(text)
        assert 151646 in text, '<|endofprompt|> not detected in CosyVoice3 text or prompt_text, check your input!'

        sos_emb: Tensor = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb: Tensor = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)
        prompt_speech_token_emb = self.speech_embedding(params.llm_prompt_speech_token)
        lm_input: Tensor = torch.concat([sos_emb, text_emb, task_id_emb, prompt_speech_token_emb], dim=1)

        min_len: int = int((text_len - params.prompt_text_len) * min_token_text_ratio)
        max_len: int = int((text_len - params.prompt_text_len) * max_token_text_ratio)
        out_tokens: list[int] = []
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(
                lm_input,
                masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                cache=cache
            )
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids: int = self.sampling_ids(
                weighted_scores=logp.squeeze(dim=0), 
                decoded_tokens=out_tokens, 
                sampling=sampling, 
                ignore_eos=True if i < min_len else False
            )
            if top_ids in self.stop_token_ids:
                break
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

    def sampling_ids(
        self,
        weighted_scores: torch.Tensor,
        decoded_tokens: list[int],
        sampling: int,
        ignore_eos: bool = True,
    ):
        if ignore_eos is True:
            weighted_scores[self.speech_token_size] = -float('inf')
        top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
        return top_ids


def make_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
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


class CosyVoiceFrontEnd:
    def __init__(self, get_tokenizer: Callable, feat_extractor: Callable, campplus_model: str, speech_tokenizer_model: str, spk2info: str = '', allowed_special: str = 'all'):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=["CPUExecutionProvider"])
        self.speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_model, sess_options=option,
                                                                     providers=["CUDAExecutionProvider" if torch.cuda.is_available() else
                                                                                "CPUExecutionProvider"])
        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.device, weights_only=True)
        else:
            self.spk2info = {}
        self.allowed_special = allowed_special
        self.zh_tn_model = Normalizer(remove_erhua=False)
        self.en_tn_model = Normalizer()
        self.text_frontend = 'wetext'

    def _extract_text_token(self, text: str) -> tuple[Tensor, Tensor]:
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
        return text_token, text_token_len

    def _extract_speech_token(self, prompt_wav: str) -> tuple[Tensor, Tensor]:
        speech = load_wav(prompt_wav, 16000)
        assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(None,
                                                         {self.speech_tokenizer_session.get_inputs()[0].name:
                                                          feat.detach().cpu().numpy(),
                                                          self.speech_tokenizer_session.get_inputs()[1].name:
                                                          np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, prompt_wav):
        speech = load_wav(prompt_wav, 16000)
        feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None, {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, prompt_wav):
        speech = load_wav(prompt_wav, 24000)
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    def text_normalize(self, text: str, split: bool=True, text_frontend: bool=True):
        # NOTE skip text_frontend when ssml symbol in text
        if '<|' in text and '|>' in text:
            text_frontend = False
        if text_frontend is False or text == '':
            return [text] if split is True else text
        text = text.strip()
        assert self.text_frontend == 'wetext'
        assert contains_chinese(text)
        if self.text_frontend == 'wetext':
            text = self.zh_tn_model.normalize(text)
        text = text.replace("\n", "")
        # 删除中文之间的空格
        text = replace_blank(text)
        # 将指数转为汉字
        text = replace_corner_mark(text)
        text = text.replace(".", "。")
        text = text.replace(" - ", "，")
        text = remove_bracket(text)
        # 将文本末尾连续的逗号/顿号统一替换为一个句号
        text = re.sub(r'[，,、]+$', '。', text)
        texts = list(split_paragraph(text, tokenize=partial(self.tokenizer.encode, allowed_special=self.allowed_special), lang="zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False))
        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts if split is True else text

    def frontend_zero_shot(self, tts_text: str, prompt_text: str, prompt_wav: str, resample_rate: float, zero_shot_spk_id: str) -> dict[str, Tensor]:
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        if zero_shot_spk_id == '':
            prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
            speech_feat, speech_feat_len = self._extract_speech_feat(prompt_wav)
            speech_token, speech_token_len = self._extract_speech_token(prompt_wav)
            if resample_rate == 24000:
                # cosyvoice2, force speech_feat % speech_token = 2
                token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
                speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
                speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
            embedding = self._extract_spk_embedding(prompt_wav)
            model_input = {'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                           'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                           'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                           'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                           'llm_embedding': embedding, 'flow_embedding': embedding}
        else:
            model_input = {**self.spk2info[zero_shot_spk_id]}
        model_input['text'] = tts_text_token
        model_input['text_len'] = tts_text_token_len
        return model_input


def get_config(model_dir: str) -> dict:
    qwen_pretrain_path = os.path.join(model_dir, 'CosyVoice-BlankEN')
    sample_rate = 24000
    llm_input_size = 896
    llm_output_size = 896
    spk_embed_dim = 192
    token_mel_ratio = 2
    chunk_size = 25
    num_decoding_left_chunks = -1
    return {
        'sample_rate': sample_rate,
        'allowed_special': 'all',
        'get_tokenizer': partial(get_qwen_tokenizer,
                                 token_path=qwen_pretrain_path,
                                 skip_special_tokens=True,
                                 version='cosyvoice3'),
        'feat_extractor': partial(mel_spectrogram,
                                  n_fft=1920, num_mels=80,
                                  sampling_rate=sample_rate,
                                  hop_size=480, win_size=1920,
                                  fmin=0, fmax=None, center=False),
        'llm': CosyVoice3LM(
            llm_input_size=llm_input_size,
            llm_output_size=llm_output_size,
            speech_token_size=6561,
            llm=Qwen2Encoder(pretrain_path=qwen_pretrain_path),
            sampling=partial(ras_sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1),
        ),
        'flow': CausalMaskedDiffWithDiT(
            input_size=80,
            output_size=80,
            spk_embed_dim=spk_embed_dim,
            vocab_size=6561,
            token_mel_ratio=token_mel_ratio,
            pre_lookahead_len=3,
            pre_lookahead_layer=PreLookaheadLayer(
                in_channels=80, channels=1024, pre_lookahead_len=3,
            ),
            decoder=CausalConditionalCFM(
                in_channels=240,
                n_spks=1,
                spk_emb_dim=80,
                t_scheduler='cosine',
                inference_cfg_rate=0.7,
                estimator=DiT(
                    dim=1024, depth=22, heads=16, dim_head=64,
                    ff_mult=2, mel_dim=80, mu_dim=80, spk_dim=80,
                    out_channels=80,
                    static_chunk_size=chunk_size * token_mel_ratio,
                    num_decoding_left_chunks=num_decoding_left_chunks,
                ),
            ),
        ),
        'hift': CausalHiFTGenerator(
            in_channels=80,
            base_channels=512,
            nb_harmonics=8,
            sampling_rate=sample_rate,
            nsf_alpha=0.1,
            nsf_sigma=0.003,
            nsf_voiced_threshold=10,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            istft_params={'n_fft': 16, 'hop_len': 4},
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            lrelu_slope=0.1,
            audio_limit=0.99,
            conv_pre_look_right=4,
            f0_predictor=CausalConvRNNF0Predictor(
                num_class=1, in_channels=80, cond_channels=512,
            ),
        ),
    }


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

    def forward(self, x: Tensor, emb: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        # x (b, n, d) mask (b, 1, n, n) rope (1, n, d)
        batch_size = x.shape[0]
        query, key, value = self.to_q(x), self.to_k(x), self.to_v(x)
        freqs, scale = rope
        query, key = apply_rotary_pos_emb(query, freqs), apply_rotary_pos_emb(key, freqs)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (b, n, d) -> (b, h, n, d)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (b, n, d) -> (b, h, n, d)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (b, n, d) -> (b, h, n, d)

        x = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim) # (b, h, n, d) -> (b, n, h * d)
        x = x.to(query.dtype)
        # linear proj
        x = self.to_out[0](x)

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
        # x (b, n, d) t (b, d) mask (b, 1, n, n) rope (1, n, head_dim=64)
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t) # (b, n, d) (b, d) (b, d) (b, d) (b, d)
        attn_output = self.attn(x=norm, mask=mask, rope=rope) # (b, n, d)
        x = x + gate_msa.unsqueeze(1) * attn_output # (b, n, d)
        ff_norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None] # (b, n, d)
        ff_output = self.ff(ff_norm) # (b, n, d)
        x = x + gate_mlp.unsqueeze(1) * ff_output # (b, n, d)
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
    def __init__(self,  dim: int, depth: int, heads: int, dim_head: int, ff_mult: int, mel_dim: int, mu_dim: int, spk_dim: int, out_channels: int, static_chunk_size: int, num_decoding_left_chunks: int):
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
        seq_len = x.shape[1] # mel_timesteps
        t = self.time_embed(t) # (batch_size, hidden_size=1024)
        x = self.input_embed(x, cond, mu, spks.squeeze(1)) # (batch_size, mel_timesteps, hidden_size=1024)
        rope: tuple[Tensor, float] = self.rotary_embed.forward_from_seq_len(seq_len) # (1, mel_timesteps, 64)
        attn_mask: Tensor = mask.bool().repeat(1, x.size(1), 1).unsqueeze(dim=1) # (batch_size, 1, mel_timesteps) -> (batch_size, mel_timesteps, mel_timesteps) -> (batch_size, 1, mel_timesteps, mel_timesteps)
        for block in self.transformer_blocks:
            x = block(x, t, mask=attn_mask, rope=rope) # (batch_size, mel_timesteps, hidden_size=1024)
        x = self.norm_out(x, t) # (batch_size, mel_timesteps, hidden_size=1024)
        return self.proj_out(x).transpose(1, 2) # (batch_size, hidden_size=80, mel_timesteps)


class PreLookaheadLayer(nn.Module):
    def __init__(self, in_channels: int, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            in_channels, channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1, padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels, in_channels,
            kernel_size=3, stride=1, padding=0,
        )

    def forward(self, inputs: torch.Tensor, context: torch.Tensor = torch.zeros(0, 0, 0)) -> torch.Tensor:
        # inputs: (batch_size, seq_len, channels)
        outputs = inputs.transpose(1, 2).contiguous()
        context = context.transpose(1, 2).contiguous()
        outputs = F.pad(input=outputs, pad=(0, self.pre_lookahead_len), mode='constant', value=0.0)
        outputs = F.leaky_relu(input=self.conv1(outputs))
        outputs = F.pad(input=outputs, pad=(self.conv2.kernel_size[0] - 1, 0), mode='constant', value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()
        outputs = outputs + inputs
        return outputs


class CausalMaskedDiffWithDiT(torch.nn.Module):
    def __init__(
        self,
        pre_lookahead_layer: torch.nn.Module, 
        decoder: torch.nn.Module, 
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        vocab_size: int = 4096,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.input_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.pre_lookahead_len = pre_lookahead_len
        self.pre_lookahead_layer = pre_lookahead_layer
        self.decoder = decoder
        self.token_mel_ratio = token_mel_ratio


    @torch.inference_mode()
    def inference(self, params: FlowInputParams):
        assert params.token.shape[0] == 1
        print_params(params, "Flow Model Input")
        embedding = F.normalize(input=params.embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        token, token_len = torch.concat([params.prompt_token, params.token], dim=1), params.prompt_token_len + params.token_len # (batch_size, n_tokens=n_prompt_tokens + n_tts_speech_tokens) batch_size must be 1
        mask = (~make_pad_mask(token_len)).unsqueeze(-1) # (batch_size, max(n_tokens), 1) value 0 is masked
        token = self.input_embedding(torch.clamp(token, min=0)) * mask # (batch_size, t=max(n_tokens), hidden_size=80)

        if params.finalize is True:
            h = self.pre_lookahead_layer(token) # (batch_size, t, hidden_size=80)
        else:
            h = self.pre_lookahead_layer(token[:, :-self.pre_lookahead_len], context=token[:, -self.pre_lookahead_len:])
        h = h.repeat_interleave(self.token_mel_ratio, dim=1) # (batch_size, mel_timesteps=t * token_mel_ratio=2t, hidden_size=80)
        mu = h.transpose(1, 2).contiguous()
        mel_len1, mel_len2 = params.prompt_feat.shape[1], h.shape[1] - params.prompt_feat.shape[1]

        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], dtype=h.dtype) # (batch_size, mel_timesteps, hidden_size=80) batch_size must be 1
        conds[:, :mel_len1] = params.prompt_feat
        conds = conds.transpose(1, 2) # (batch_size, hidden_size=80, mel_timesteps)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).unsqueeze(1) # batch_size must be 1
        feat = self.decoder(
            mu=mu,
            mask=mask,
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=params.streaming
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float()


class CausalConditionalCFM(nn.Module):
    def __init__(self, in_channels, t_scheduler='cosine', inference_cfg_rate=0.7, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__()
        self.rand_noise = torch.randn([1, 80, 50 * 300])
        self.t_scheduler = t_scheduler
        self.inference_cfg_rate = inference_cfg_rate
        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu: Tensor, mask: Tensor, n_timesteps: int, temperature: float=1.0, spks: Tensor | None=None, cond: Tensor | None=None, streaming: bool=False) -> Tensor:
        # mu (batch_size, hidden_size, mel_timesteps)
        # mask (batch_size, 1, mel_timesteps)
        # spks shape: (batch_size, hidden_size)
        # cond (batch_size, hidden_size, mel_timesteps)
        x: Tensor = self.rand_noise[:, :, :mu.size(2)].to(mu.device).to(mu.dtype) * temperature
        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype) # (n_timesteps + 1,)
        if self.t_scheduler == 'cosine':
            t_span: Tensor = 1 - torch.cos(t_span * 0.5 * torch.pi)

        # generated mel-spectrogram (batch_size, n_feats, mel_timesteps)
        t, dt = t_span[0], t_span[1] - t_span[0]

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        x_in: Tensor = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        mask_in: Tensor = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=spks.dtype)
        mu_in: Tensor = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        t_in: Tensor = torch.zeros([2], device=x.device, dtype=spks.dtype)
        spks_in: Tensor = torch.zeros([2, 80], device=x.device, dtype=spks.dtype)
        cond_in: Tensor = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        for step in range(1, n_timesteps + 1):
            x_in[:], mask_in[:], t_in[:] = x, mask, t
            mu_in[0], spks_in[0], cond_in[0] = mu, spks, cond
            dphi_dt = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming)
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0) # (1, hidden_size, mel_timesteps) (1, hidden_size, mel_timesteps)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x: Tensor = x + dt * dphi_dt
            t = t + dt
            if step < n_timesteps:
                dt = t_span[step + 1] - t

        return x.float()


class Snake(nn.Module):
    def __init__(self, in_features: int, alpha: float=1.0):
        super(Snake, self).__init__()
        self.alpha = nn.Parameter(torch.ones(in_features) * alpha, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        # Snake ∶= x + 1/a * sin^2 (xa)
        # x (B, C, T)
        alpha: Tensor = self.alpha.unsqueeze(dim=0).unsqueeze(-1)  # (C, 1) -> (B, C, T)
        x = x + (1.0 / (alpha + 0.000000001)) * pow(sin(x * alpha), 2)
        return x


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        causal_type: Literal['left', 'right'] = 'left',
        device=None,
        dtype=None
    ) -> None:
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        assert stride == 1
        self.causal_padding: int = int((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2
        self.causal_type: str = causal_type


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        causal_type: Literal['left', 'right'] = 'left',
        device=None,
        dtype=None
    ) -> None:
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        assert stride == 1
        self.causal_padding: int = int((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2
        self.causal_type: str = causal_type

    def forward(self, x: Tensor, cache: Tensor | None = None) -> Tensor:
        # x (B, C, T) cache (B, C, causal_padding)
        input_timestep = x.shape[2]
        if cache is None:
            cache = torch.zeros(x.shape[0], x.shape[1], self.causal_padding, dtype=x.dtype, device=x.device)
        if self.causal_type == 'left':
            x = torch.concat([cache, x], dim=2)
        else:
            x = torch.concat([x, cache], dim=2)
        x = super(CausalConv1d, self).forward(x)
        assert x.shape[2] == input_timestep
        return x
    def forward(self, x: Tensor, cache: Tensor | None = None) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        # x (B, C, T) cache (B, C, causal_padding)
        if cache is None:
            cache = torch.zeros(x.shape[0], x.shape[1], self.causal_padding, dtype=x.dtype, device=x.device)
        o = torch.concat([cache, x] if self.causal_type == 'left' else [x, cache], dim=2)
        o = super(CausalConv1d, self).forward(o)
        assert o.shape[2] == x.shape[2] # not pass if kernel_size and dilation are both even
        return o


class CausalConv1dDownSample(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        super(CausalConv1dDownSample, self).__init__(in_channels, out_channels,
                                                     kernel_size, stride,
                                                     padding=0, dilation=dilation,
                                                     groups=groups, bias=bias,
                                                     padding_mode=padding_mode,
                                                     device=device, dtype=dtype)
        assert stride != 1 and dilation == 1
        assert kernel_size % stride == 0
        self.causal_padding = stride - 1

    def forward(self, x: torch.Tensor, cache: torch.Tensor = torch.zeros(0, 0, 0)) -> Tuple[torch.Tensor, torch.Tensor]:
        if cache.size(2) == 0:
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            assert cache.size(2) == self.causal_padding
            x = torch.concat([cache, x], dim=2)
        x = super(CausalConv1dDownSample, self).forward(x)
        return x


class CausalConv1dUpsample(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        super(CausalConv1dUpsample, self).__init__(in_channels, out_channels,
                                                   kernel_size, 1,
                                                   padding=0, dilation=dilation,
                                                   groups=groups, bias=bias,
                                                   padding_mode=padding_mode,
                                                   device=device, dtype=dtype)
        assert dilation == 1
        self.causal_padding = kernel_size - 1
        self.upsample = torch.nn.Upsample(scale_factor=stride, mode='nearest')

    def forward(self, x: torch.Tensor, cache: torch.Tensor = torch.zeros(0, 0, 0)) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.upsample(x)
        input_timestep = x.shape[2]
        if cache.size(2) == 0:
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            assert cache.size(2) == self.causal_padding
            x = torch.concat([cache, x], dim=2)
        x = super(CausalConv1dUpsample, self).forward(x)
        assert input_timestep == x.shape[2]
        return x


class ResBlock(torch.nn.Module):
    def __init__(self, channels: int = 512, kernel_size: int = 3, dilations: list[int] = [1, 3, 5], causal: bool = False):
        super(ResBlock, self).__init__()
        self.causal = causal
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for dilation in dilations:
            _ = self.convs1.append(weight_norm(CausalConv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                causal_type='left'
            )))
            _ = self.convs2.append(weight_norm(CausalConv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=1,
                causal_type='left'
            )))
        self.activations1 = nn.ModuleList([Snake(channels) for _ in range(len(self.convs1))])
        self.activations2 = nn.ModuleList([Snake(channels) for _ in range(len(self.convs2))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt)
            x = xt + x
        return x


class SineGen2(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """
    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False,
                 causal=False):
        super(SineGen2, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale
        self.causal = causal
        if causal is True:
            self.rand_ini = torch.rand(1, 9)
            self.rand_ini[:, 0] = 0
            self.sine_waves = torch.rand(1, 300 * 24000, 9)

    def _f02uv(self, f0):
        # generate uv (unvoiced voiced) signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values: Tensor) -> Tensor:
        # f0_values: (batchsize, length, dim) where dim indicates fundamental tone and overtones
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1
        # initial phase noise (no noise for fundamental component)
        assert self.training is False and self.causal is True
        rad_values[:, 0, :] = rad_values[:, 0, :] + self.rand_ini.to(rad_values.device)
        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        assert not self.flag_for_pulse
        rad_values = torch.nn.functional.interpolate(rad_values.transpose(1, 2),
                                                        scale_factor=1 / self.upsample_scale,
                                                        mode="linear").transpose(1, 2)

        phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
        phase = torch.nn.functional.interpolate(phase.transpose(1, 2) * self.upsample_scale,
                                                scale_factor=self.upsample_scale, mode="nearest" if self.causal is True else 'linear').transpose(1, 2)
        sines = torch.sin(phase)
        return sines

    def forward(self, f0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        # fundamental component
        fn: Tensor = f0 * torch.arange(1, self.harmonic_num + 2, dtype=f0.dtype, device=f0.device) # (batch_size=1, length, dim=harmonic_num + 2)
        # generate sine waveforms
        sine_waves: Tensor = self._f02sine(fn) * self.sine_amp # (batch_size=1, length, dim=harmonic_num + 2)
        # generate uv signal
        uv: Tensor = self._f02uv(f0)  # (batch_size=1, length, dim=1)
        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        # .       for voiced regions is self.noise_std
        noise_amp: Tensor = uv * self.noise_std + (1 - uv) * self.sine_amp / 3 # (batch_size=1, length, dim=1)
        assert self.training is False and self.causal is True
        noise: Tensor = noise_amp * self.sine_waves[:, :sine_waves.shape[1]].to(sine_waves.device)
        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves: Tensor = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0, sinegen_type='1', causal=False):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        # to produce sine waveforms
        assert sinegen_type != '1'
        self.l_sin_gen = SineGen2(sampling_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshod, causal=causal)
        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()
        self.causal = causal
        if causal is True:
            self.uv = torch.rand(1, 300 * 24000, 1)

    def forward(self, x: Tensor):
        sine_wavs, uv, _ = self.l_sin_gen(x) # (batch_size=1, length, dim=harmonic_num + 2)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs)) # (batch_size=1, length, dim=1)
        assert self.training is False and self.causal is True
        noise = self.uv[:, :uv.shape[1]] * self.sine_amp / 3
        return sine_merge, noise, uv


class CausalHiFTGenerator(nn.Module):
    def __init__(
            self,
            in_channels: int = 80,
            base_channels: int = 512,
            nb_harmonics: int = 8,
            sampling_rate: int = 22050,
            nsf_alpha: float = 0.1,
            nsf_sigma: float = 0.003,
            nsf_voiced_threshold: float = 10,
            upsample_rates: list[int] = [8, 8],
            upsample_kernel_sizes: list[int] = [16, 16],
            istft_params: dict[str, int] = {"n_fft": 16, "hop_len": 4},
            resblock_kernel_sizes: list[int] = [3, 7, 11],
            resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes: list[int] = [7, 11],
            source_resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5]],
            lrelu_slope: float = 0.1,
            audio_limit: float = 0.99,
            conv_pre_look_right: int = 4,
            f0_predictor: torch.nn.Module = None,
    ):
        torch.nn.Module.__init__(self)

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=np.prod(upsample_rates) * istft_params["hop_len"],
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshod=nsf_voiced_threshold,
            sinegen_type='1' if self.sampling_rate == 22050 else '2',
            causal=True)
        self.upsample_rates = upsample_rates
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates) * istft_params["hop_len"]) # upsample_rates = [8, 5, 3] istft_params["hop_len"] = 4

        self.conv_pre = weight_norm(
            CausalConv1d(in_channels, base_channels, conv_pre_look_right + 1, 1, causal_type='right')
        )

        # Up
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(CausalConv1dUpsample(base_channels // (2**i),base_channels // (2**(i + 1)),k,u)))

        # Down
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes)):
            if u == 1:
                self.source_downs.append(CausalConv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), 1, 1, causal_type='left'))
            else:
                self.source_downs.append(CausalConv1dDownSample(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), u * 2, u))

            self.source_resblocks.append(ResBlock(base_channels // (2 ** (i + 1)), k, d, causal=True))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2**(i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d, causal=True))

        self.conv_post = weight_norm(CausalConv1d(ch, istft_params["n_fft"] + 2, 7, 1, causal_type='left'))
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft_window = torch.from_numpy(get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32))
        self.conv_pre_look_right = conv_pre_look_right
        self.f0_predictor = f0_predictor

    def _stft(self, x):
        spec = torch.stft(
            x,
            self.istft_params["n_fft"], self.istft_params["hop_len"], self.istft_params["n_fft"], window=self.stft_window.to(x.device),
            return_complex=True)
        spec = torch.view_as_real(spec)  # [B, F, TT, 2]
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        inverse_transform = torch.istft(torch.complex(real, img), self.istft_params["n_fft"], self.istft_params["hop_len"],
                                        self.istft_params["n_fft"], window=self.stft_window.to(magnitude.device))
        return inverse_transform

    def decode(self, x: torch.Tensor, s: torch.Tensor = torch.zeros(1, 1, 0), finalize: bool = True) -> torch.Tensor:
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        if finalize is True:
            x = self.conv_pre(x)
        else:
            x = self.conv_pre(x[:, :, :-self.conv_pre_look_right], x[:, :, -self.conv_pre_look_right:])
            s_stft_real = s_stft_real[:, :, :-int(np.prod(self.upsample_rates) * self.conv_pre_look_right)]
            s_stft_imag = s_stft_imag[:, :, :-int(np.prod(self.upsample_rates) * self.conv_pre_look_right)]
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # fusion
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, :self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1:, :])  # actually, sin is redundancy

        x = self._istft(magnitude, phase)
        if finalize is False:
            x = x[:, :-int(np.prod(self.upsample_rates) * self.istft_params['hop_len'])]
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    @torch.inference_mode()
    def inference(self, speech_feat: torch.Tensor, finalize: bool = True) -> torch.Tensor:
        # mel->f0 NOTE f0_predictor precision is crucial for causal inference, move self.f0_predictor to cpu if necessary
        self.f0_predictor.to(torch.float64)
        f0 = self.f0_predictor(speech_feat.to(torch.float64), finalize=finalize).to(speech_feat)
        # f0->source
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        assert finalize is True
        if finalize is True:
            generated_speech = self.decode(x=speech_feat, s=s, finalize=finalize)
        else:
            generated_speech = self.decode(x=speech_feat[:, :, :-self.f0_predictor.condnet[0].causal_padding], s=s, finalize=finalize)
        return generated_speech, s


class CausalConvRNNF0Predictor(nn.Module):
    def __init__(self, num_class: int = 1, in_channels: int = 80, cond_channels: int = 512):
        super().__init__()
        self.condnet = nn.Sequential(
            weight_norm(CausalConv1d(in_channels, cond_channels, kernel_size=4, causal_type='right')),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type='left')),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type='left')),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type='left')),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type='left')),
            nn.ELU(),
        )
        self.classifier = nn.Linear(in_features=cond_channels, out_features=num_class)

    def forward(self, x: torch.Tensor, finalize: bool = True) -> torch.Tensor:
        if finalize is True:
            x = self.condnet[0](x)
        else:
            x = self.condnet[0](x[:, :, :-self.condnet[0].causal_padding], x[:, :, -self.condnet[0].causal_padding:])
        for i in range(1, len(self.condnet)):
            x = self.condnet[i](x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


class CosyVoice3:
    def __init__(self, model_dir: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.silent_tokens: list[int] = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]

        configs = get_config(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          f'{model_dir}/campplus.onnx',
                                          f'{model_dir}/speech_tokenizer_v3.onnx',
                                          f'{model_dir}/spk2info.pt',
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        self.llm, self.flow, self.hift = configs['llm'], configs['flow'], configs['hift']
        del configs

        self.llm.load_state_dict(torch.load(f'{model_dir}/llm.pt', map_location=self.device, weights_only=True), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(f'{model_dir}/flow.pt', map_location=self.device, weights_only=True), strict=True)
        self.flow.to(self.device).eval()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(f'{model_dir}/hift.pt', map_location=self.device, weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def tts(self, params: TTSInputParams) -> Generator[dict[str, Tensor], None, None]:
        print_params(params, "TTS Input")
        assert params.source_speech_token.shape[1] == 0
        assert params.stream is False
        with Timer("llm"):
            tts_speech_token: list[int] = []
            cur_silent_token_num, max_silent_token_num = 0, 5
            token_generator = self.llm.inference(params)
            for i in token_generator:
                if i in self.silent_tokens:
                    cur_silent_token_num += 1
                    if cur_silent_token_num > max_silent_token_num:
                        continue
                else:
                    cur_silent_token_num = 0
                tts_speech_token.append(i)
            this_tts_speech_token: Tensor = torch.tensor(tts_speech_token).unsqueeze(dim=0)
            this_tts_speech_token_len: Tensor = torch.tensor([this_tts_speech_token.shape[1]], dtype=torch.int32)
        with Timer("flow"):
            tts_mel = self.flow.inference(FlowInputParams(
                token=this_tts_speech_token, 
                token_len=this_tts_speech_token_len, 
                prompt_token=params.flow_prompt_speech_token, 
                prompt_token_len=params.flow_prompt_speech_token_len, 
                prompt_feat=params.prompt_speech_feat,
                prompt_feat_len=params.prompt_speech_feat_len,
                embedding=params.flow_embedding,
                streaming=False,
                finalize=True
            ))
        with Timer("hift"):
            if params.speed != 1.0:
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / params.speed), mode='linear')
            tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=True)

        yield {'tts_speech': tts_speech.cpu()}

    def inference_zero_shot(self, tts_text: str, prompt_text: str, prompt_wav: str, zero_shot_spk_id: str='', stream: bool=False, speed: float=1.0, text_frontend=True):
        prompt_text: str = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        print(f'prompt_text {prompt_text}')
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            print(f'tts_text {tts_text}')
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_wav, self.sample_rate, zero_shot_spk_id)
            print('synthesis text {}'.format(i))
            for model_output in self.tts(TTSInputParams(**model_input, stream=stream, speed=speed)):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                yield model_output