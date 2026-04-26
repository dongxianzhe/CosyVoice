from dataclasses import dataclass, field
import torch
from torch import Tensor

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

    def print(self):
        print(f'TTSInputParams: ')
        for field, value in self.__dict__.items():
            if isinstance(value, Tensor) and value.numel() <= 16:
                print(f"    {field}: shape = {value.shape} value = {value}")
            elif isinstance(value, Tensor):
                print(f"    {field}: shape = {value.shape}")
            else:
                print(f"    {field}: {value}")


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

    def print(self):
        print(f'FlowInputParams: ')
        for field, value in self.__dict__.items():
            if isinstance(value, Tensor) and value.numel() <= 16:
                print(f"    {field}: shape = {value.shape} value = {value}")
            elif isinstance(value, Tensor):
                print(f"    {field}: shape = {value.shape}")
            else:
                print(f"    {field}: {value}")
