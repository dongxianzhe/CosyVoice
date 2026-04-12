from torch._tensor import Tensor
from typing import Generator
from torch import Tensor
import torch
from torch.nn import functional as F
from cosyvoice.model import TTSInputParams


class CosyVoice3Model:
    def __init__(self, llm: torch.nn.Module, flow: torch.nn.Module, hift: torch.nn.Module) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm, self.flow, self.hift = llm, flow, hift
        self.token_hop_len = 25
        self.token_max_hop_len = 4 * self.token_hop_len
        self.stream_scale_factor = 2
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.silent_tokens: list[int] = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]

    def load(self, llm_model: str, flow_model: str, hift_model: str) -> None:
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device, weights_only=True), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device, weights_only=True), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device, weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def tts(self, params: TTSInputParams) -> Generator[dict[str, Tensor], None, None]:
        assert params.source_speech_token.shape[1] == 0
        assert params.stream is False
        # 1. LLM 生成 speech tokens (串行)
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

        # 2. Flow + HiFT 生成波形 (串行)
        this_tts_speech_token: Tensor = torch.tensor(tts_speech_token).unsqueeze(dim=0)
        tts_mel, _ = self.flow.inference(
            token=this_tts_speech_token.to(self.device, dtype=torch.int32),
            token_len=torch.tensor([this_tts_speech_token.shape[1]], dtype=torch.int32).to(self.device),
            prompt_token=params.flow_prompt_speech_token, 
            prompt_token_len=params.flow_prompt_speech_token_len, 
            prompt_feat=params.prompt_speech_feat,
            prompt_feat_len=params.prompt_speech_feat_len,
            embedding=params.flow_embedding,
            streaming=False,
            finalize=True
        )
        if params.speed != 1.0:
            tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / params.speed), mode='linear')
        tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=True)

        yield {'tts_speech': tts_speech.cpu()}