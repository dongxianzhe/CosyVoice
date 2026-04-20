from typing import Any
import os
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from cosyvoice.utils.mask import make_pad_mask
from cosyvoice.utils.onnx import SpeechTokenExtractor, online_feature, onnx_path
from cosyvoice.model import FlowInputParams


class CausalMaskedDiffWithDiT(torch.nn.Module):
    def __init__(
        self,
        pre_lookahead_layer: torch.nn.Module, 
        decoder: torch.nn.Module, 
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        only_mask_loss: bool = True,
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
        if online_feature is True:
            self.speech_token_extractor = SpeechTokenExtractor(model_path=os.path.join(onnx_path, 'speech_tokenizer_v3.batch.onnx'))


    @torch.inference_mode()
    def inference(self, params: FlowInputParams):
        assert params.token.shape[0] == 1
        params.print()
        embedding = F.normalize(input=params.embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        token, token_len = torch.concat([params.prompt_token, params.token], dim=1), params.prompt_token_len + params.token_len # (batch_size, n_tokens=n_prompt_tokens + n_tts_speech_tokens) batch_size must be 1
        mask = (~make_pad_mask(token_len)).unsqueeze(-1) # (batch_size, max(n_tokens), 1) value 0 is masked
        token = self.input_embedding(torch.clamp(token, min=0)) * mask # (batch_size, t=max(n_tokens), hidden_size=80)

        if params.finalize is True:
            h = self.pre_lookahead_layer(token) # (batch_size, t, hidden_size=80)
        else:
            h = self.pre_lookahead_layer(token[:, :-self.pre_lookahead_len], context=token[:, -self.pre_lookahead_len:])
        h = h.repeat_interleave(self.token_mel_ratio, dim=1) # (batch_size, t * token_mel_ratio=2t, hidden_size=80)
        mel_len1, mel_len2 = params.prompt_feat.shape[1], h.shape[1] - params.prompt_feat.shape[1]

        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], dtype=h.dtype) # (batch_size, t, hidden_size=80) batch_size must be 1
        conds[:, :mel_len1] = params.prompt_feat
        conds = conds.transpose(1, 2) # (batch_size, hidden_size=80, t)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))) # batch_size must be 1
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=params.streaming
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float()