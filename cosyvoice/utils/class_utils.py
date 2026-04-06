import torch


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
}

COSYVOICE_ATTENTION_CLASSES = {
}


def get_model_type(configs):
    # NOTE CosyVoice2Model inherits CosyVoiceModel
    if isinstance(configs['llm'], CosyVoice3LM) and isinstance(configs['flow'], CausalMaskedDiffWithDiT) and isinstance(configs['hift'], CausalHiFTGenerator):
        return CosyVoice3Model
    raise TypeError('No valid model type found!')
