from typing import Callable
import torch
from torch import nn, Tensor
from transformers import Qwen2ForCausalLM
from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.model import TTSInputParams


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
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            mix_ratio: list[int] = [5, 15],
    ):
        torch.nn.Module.__init__(self)
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        # 2. build speech token language model related modules
        self.sos: int = speech_token_size + 0
        self.eos_token: int = speech_token_size + 1
        self.task_id: int = speech_token_size + 2
        self.fill_token: int = speech_token_size + 3
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 200, bias=False)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 200,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 200, llm_input_size)
        self.sampling = sampling
        self.mix_ratio = mix_ratio
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