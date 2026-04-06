import torch


from cosyvoice.transformer.embedding import (PositionalEncoding,
                                             RelPositionalEncoding,
                                             WhisperPositionalEncoding,
                                             LearnablePositionalEncoding,
                                             NoPositionalEncoding)
from cosyvoice.transformer.attention import (MultiHeadedAttention,
                                             RelPositionMultiHeadedAttention)
from cosyvoice.transformer.embedding import EspnetRelPositionalEncoding
from cosyvoice.llm.llm import CosyVoice3LM
from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
from cosyvoice.hifigan.generator import CausalHiFTGenerator
from cosyvoice.cli.model import CosyVoice3Model


COSYVOICE_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "gelu": torch.nn.GELU,
}

COSYVOICE_SUBSAMPLE_CLASSES = {
    'paraformer_dummy': torch.nn.Identity
}

COSYVOICE_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "rel_pos_espnet": EspnetRelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
}

COSYVOICE_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
}


def get_model_type(configs):
    # NOTE CosyVoice2Model inherits CosyVoiceModel
    if isinstance(configs['llm'], CosyVoice3LM) and isinstance(configs['flow'], CausalMaskedDiffWithDiT) and isinstance(configs['hift'], CausalHiFTGenerator):
        return CosyVoice3Model
    raise TypeError('No valid model type found!')
