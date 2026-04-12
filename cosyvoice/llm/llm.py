from typing import Any, Callable, List, Generator
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
            mix_ratio: List[int] = [5, 15],
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
    def inference(self, params: TTSInputParams, sampling: int = 25, max_token_text_ratio: float = 20, min_token_text_ratio: float = 2) -> Generator[torch.Tensor, None, None]:
        text = torch.concat([params.prompt_text, params.text], dim=1)
        params.text_len += params.prompt_text_len
        text_emb = self.llm.model.model.embed_tokens(text)
        assert 151646 in text, '<|endofprompt|> not detected in CosyVoice3 text or prompt_text, check your input!'

        sos_emb: Tensor = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb: Tensor = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)
        if params.llm_prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(params.llm_prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text_emb.dtype)
        lm_input: Tensor = torch.concat([sos_emb, text_emb, task_id_emb, prompt_speech_token_emb], dim=1)

        min_len: int = int((params.text_len - params.prompt_text_len) * min_token_text_ratio)
        max_len: int = int((params.text_len - params.prompt_text_len) * max_token_text_ratio)
        for token in self.inference_wrapper(lm_input, sampling, min_len, max_len):
            yield token

    @torch.inference_mode()
    def inference_wrapper(self, lm_input: Tensor, sampling: int, min_len: int, max_len: int) -> Generator[Any, Any, None]:
        out_tokens = []
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                        masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                        cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False)
            if top_ids in self.stop_token_ids:
                break
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

    def sampling_ids(
        self,
        weighted_scores: torch.Tensor,
        decoded_tokens: List,
        sampling: int,
        ignore_eos: bool = True,
    ):
        if ignore_eos is True:
            weighted_scores[self.speech_token_size] = -float('inf')
        top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
        return top_ids