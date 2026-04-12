from torch import Tensor
import torch
import threading
from torch.nn import functional as F
from contextlib import nullcontext
import uuid


class CosyVoice3Model:
    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 ) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        # NOTE must matching training static_chunk_size
        self.token_hop_len = 25
        # NOTE increase token_hop_len incrementally to avoid duplicate inference
        self.token_max_hop_len = 4 * self.token_hop_len
        self.stream_scale_factor = 2
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        # rtf and decoding related
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}
        self.silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]

    def token2wav(self, token: Tensor, prompt_token: Tensor, prompt_feat: Tensor, embedding: Tensor, token_offset, uuid, stream=False, finalize=False, speed: float=1.0):
        tts_mel, _ = self.flow.inference(token=token.to(self.device, dtype=torch.int32),
                                            token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_token=prompt_token.to(self.device),
                                            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_feat=prompt_feat.to(self.device),
                                            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                            embedding=embedding.to(self.device),
                                            streaming=stream,
                                            finalize=finalize)
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append mel cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel = self.hift_cache_dict[uuid]['mel']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
            self.hift_cache_dict[uuid]['mel'] = tts_mel
        else:
            self.hift_cache_dict[uuid] = {'mel': tts_mel, 'speech_offset': 0}
        if speed != 1.0:
            assert token_offset == 0 and finalize is True, 'speed change only support non-stream inference mode'
            tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
        tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)
        tts_speech = tts_speech[:, self.hift_cache_dict[uuid]['speech_offset']:]
        self.hift_cache_dict[uuid]['speech_offset'] += tts_speech.shape[1]
        return tts_speech

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device, weights_only=True), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device, weights_only=True), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device, weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        cur_silent_token_num, max_silent_token_num = 0, 5
        token_generator = self.llm.inference(text=text.to(self.device),
                                                text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                                prompt_text=prompt_text.to(self.device),
                                                prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                embedding=llm_embedding.to(self.device),
                                                uuid=uuid)
        for i in token_generator:
            if i in self.silent_tokens:
                cur_silent_token_num += 1
                if cur_silent_token_num > max_silent_token_num:
                    continue
            else:
                cur_silent_token_num = 0
            self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
        assert source_speech_token.shape[1] == 0
        p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        p.start()
        assert stream is False
        # deal with all tokens
        p.join()
        this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
        this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                            prompt_token=flow_prompt_speech_token,
                                            prompt_feat=prompt_speech_feat,
                                            embedding=flow_embedding,
                                            token_offset=0,
                                            uuid=this_uuid,
                                            finalize=True,
                                            speed=speed)
        yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.current_stream().synchronize()