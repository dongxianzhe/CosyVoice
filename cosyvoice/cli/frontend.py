from torch._tensor import Tensor
from torch._tensor import Tensor


from functools import partial
from torch import Tensor
from typing import Any, Generator
import json
import onnxruntime
import torch
import numpy as np
import whisper
from typing import Callable
import torchaudio.compliance.kaldi as kaldi
import os
import re
import inflect
from cosyvoice.utils.file_utils import logging, load_wav
from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, split_paragraph, is_only_punctuation


class CosyVoiceFrontEnd:

    def __init__(self,
                 get_tokenizer: Callable,
                 feat_extractor: Callable,
                 campplus_model: str,
                 speech_tokenizer_model: str,
                 spk2info: str = '',
                 allowed_special: str = 'all'):
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
        self.inflect_parser = inflect.engine()
        from wetext import Normalizer as ZhNormalizer
        from wetext import Normalizer as EnNormalizer
        self.zh_tn_model = ZhNormalizer(remove_erhua=False)
        self.en_tn_model = EnNormalizer()
        self.text_frontend = 'wetext'
        logging.info('use wetext frontend')

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
        text = replace_blank(text)
        text = replace_corner_mark(text)
        text = text.replace(".", "。")
        text = text.replace(" - ", "，")
        text = remove_bracket(text)
        text = re.sub(r'[，,、]+$', '。', text)
        texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False))
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