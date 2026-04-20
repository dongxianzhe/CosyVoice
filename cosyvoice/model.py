from dataclasses import dataclass, field
import torch
from torch import Tensor
# ======================== fixed params ========================

@dataclass
class GlobalConfig:
    sample_rate: int = 24000
    llm_input_size: int = 896
    llm_output_size: int = 896
    spk_embed_dim: int = 192
    qwen_pretrain_path: str = ''
    token_frame_rate: int = 25
    token_mel_ratio: int = 2


# ======================== stream related params ========================

@dataclass
class StreamConfig:
    chunk_size: int = 25                  # streaming inference chunk size, in token
    num_decoding_left_chunks: int = -1    # <0 means use all left chunks


# ======================== LLM ========================

@dataclass
class Qwen2EncoderConfig:
    """cosyvoice.llm.llm.Qwen2Encoder"""
    pretrain_path: str = ''


@dataclass
class SamplingConfig:
    """name:cosyvoice.utils.common.ras_sampling"""
    top_p: float = 0.8
    top_k: int = 25
    win_size: int = 10
    tau_r: float = 0.1


@dataclass
class LLMConfig:
    """cosyvoice.llm.llm.CosyVoice3LM"""
    llm_input_size: int = 896
    llm_output_size: int = 896
    speech_token_size: int = 6561
    length_normalized_loss: bool = True
    lsm_weight: int = 0
    mix_ratio: list[int] = field(default_factory=lambda: [5, 15])
    llm: Qwen2EncoderConfig = field(default_factory=Qwen2EncoderConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


# ======================== Flow ========================

@dataclass
class PreLookaheadLayerConfig:
    """cosyvoice.transformer.upsample_encoder.PreLookaheadLayer"""
    in_channels: int = 80
    channels: int = 1024
    pre_lookahead_len: int = 3


@dataclass
class CFMParamsConfig:
    sigma_min: float = 1e-06
    solver: str = 'euler'
    t_scheduler: str = 'cosine'
    training_cfg_rate: float = 0.2
    inference_cfg_rate: float = 0.7
    reg_loss_type: str = 'l1'


@dataclass
class DiTConfig:
    """cosyvoice.flow.DiT.dit.DiT"""
    dim: int = 1024
    depth: int = 22
    heads: int = 16
    dim_head: int = 64
    ff_mult: int = 2
    mel_dim: int = 80
    mu_dim: int = 80
    spk_dim: int = 80
    out_channels: int = 80
    static_chunk_size: int = 50    # chunk_size(25) * token_mel_ratio(2)
    num_decoding_left_chunks: int = -1


@dataclass
class CausalConditionalCFMConfig:
    """cosyvoice.flow.flow_matching.CausalConditionalCFM"""
    in_channels: int = 240
    n_spks: int = 1
    spk_emb_dim: int = 80
    cfm_params: CFMParamsConfig = field(default_factory=CFMParamsConfig)
    estimator: DiTConfig = field(default_factory=DiTConfig)


@dataclass
class FlowConfig:
    """cosyvoice.flow.flow.CausalMaskedDiffWithDiT"""
    input_size: int = 80
    output_size: int = 80
    spk_embed_dim: int = 192
    output_type: str = 'mel'
    vocab_size: int = 6561
    input_frame_rate: int = 25
    only_mask_loss: bool = True
    token_mel_ratio: int = 2
    pre_lookahead_len: int = 3
    pre_lookahead_layer: PreLookaheadLayerConfig = field(default_factory=PreLookaheadLayerConfig)
    decoder: CausalConditionalCFMConfig = field(default_factory=CausalConditionalCFMConfig)


# ======================== HiFT (vocoder) ========================

@dataclass
class ISTFTParamsConfig:
    n_fft: int = 16
    hop_len: int = 4


@dataclass
class CausalConvRNNF0PredictorConfig:
    """cosyvoice.hifigan.f0_predictor.CausalConvRNNF0Predictor"""
    num_class: int = 1
    in_channels: int = 80
    cond_channels: int = 512


@dataclass
class HiFTGeneratorConfig:
    """cosyvoice.hifigan.generator.CausalHiFTGenerator"""
    in_channels: int = 80
    base_channels: int = 512
    nb_harmonics: int = 8
    sampling_rate: int = 24000
    nsf_alpha: float = 0.1
    nsf_sigma: float = 0.003
    nsf_voiced_threshold: int = 10
    upsample_rates: list[int] = field(default_factory=lambda: [8, 5, 3])
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [16, 11, 7])
    istft_params: ISTFTParamsConfig = field(default_factory=ISTFTParamsConfig)
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    source_resblock_kernel_sizes: list[int] = field(default_factory=lambda: [7, 7, 11])
    source_resblock_dilation_sizes: list[list[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    lrelu_slope: float = 0.1
    audio_limit: float = 0.99
    conv_pre_look_right: int = 4
    f0_predictor: CausalConvRNNF0PredictorConfig = field(default_factory=CausalConvRNNF0PredictorConfig)


# ======================== GAN (HiFiGAN) ========================

@dataclass
class MelSpecTransformConfig:
    n_fft: int = 1920
    num_mels: int = 80
    sampling_rate: int = 24000
    hop_size: int = 480
    win_size: int = 1920
    fmin: int = 0
    fmax: None = None
    center: bool = False


@dataclass
class HiFiGanConfig:
    """cosyvoice.hifigan.hifigan.HiFiGan"""
    generator: HiFTGeneratorConfig = field(default_factory=HiFTGeneratorConfig)
    # discriminator 包含 mpd 和 mrd, 这里简化为标记
    # mpd: matcha.hifigan.models.MultiPeriodDiscriminator
    # mrd: cosyvoice.hifigan.discriminator.MultiResSpecDiscriminator


# ======================== Processor Functions ========================

@dataclass
class TokenizerConfig:
    token_path: str = ''
    skip_special_tokens: bool = True
    version: str = 'cosyvoice3'


@dataclass
class FilterConfig:
    max_length: int = 40960
    min_length: int = 100
    token_max_length: int = 200
    token_min_length: int = 1


@dataclass
class ResampleConfig:
    resample_rate: int = 24000


@dataclass
class TruncateConfig:
    truncate_length: int = 24480    # must be a multiplier of hop_size


@dataclass
class FeatExtractorConfig:
    """matcha.utils.audio.mel_spectrogram"""
    n_fft: int = 1920
    num_mels: int = 80
    sampling_rate: int = 24000
    hop_size: int = 480
    win_size: int = 1920
    fmin: int = 0
    fmax: None = None
    center: bool = False


@dataclass
class ComputeF0Config:
    sample_rate: int = 24000
    hop_size: int = 480


@dataclass
class ShuffleConfig:
    shuffle_size: int = 1000


@dataclass
class SortConfig:
    sort_size: int = 500    # sort_size should be less than shuffle_size


@dataclass
class BatchConfig:
    batch_type: str = 'dynamic'
    max_frames_in_batch: int = 2000


@dataclass
class PaddingConfig:
    use_spk_embedding: bool = False    # change to True during sft


# ======================== Train Config ========================

@dataclass
class OptimConfig:
    lr: float = 1e-5    # change to 1e-5 during sft


@dataclass
class SchedulerConfig:
    warmup_steps: int = 2500


@dataclass
class TrainConfig:
    optim: str = 'adam'
    optim_conf: OptimConfig = field(default_factory=OptimConfig)
    scheduler: str = 'constantlr'    # change to constantlr during sft
    scheduler_conf: SchedulerConfig = field(default_factory=SchedulerConfig)
    max_epoch: int = 200
    grad_clip: int = 5
    accum_grad: int = 2
    log_interval: int = 100
    save_per_step: int = -1


@dataclass
class OptimGanConfig:
    lr: float = 0.0002


@dataclass
class TrainGanConfig:
    optim: str = 'adam'
    optim_conf: OptimGanConfig = field(default_factory=OptimGanConfig)
    scheduler: str = 'constantlr'
    optim_d: str = 'adam'
    optim_conf_d: OptimGanConfig = field(default_factory=OptimGanConfig)
    scheduler_d: str = 'constantlr'
    max_epoch: int = 200
    grad_clip: int = 5
    accum_grad: int = 1    # in gan training, accum_grad must be 1
    log_interval: int = 100
    save_per_step: int = -1


# ======================== Top-level Config ========================

@dataclass
class CosyVoice3Config:
    """CosyVoice3 完整配置"""
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    hift: HiFTGeneratorConfig = field(default_factory=HiFTGeneratorConfig)
    hifigan: HiFiGanConfig = field(default_factory=HiFiGanConfig)
    mel_spec_transform: MelSpecTransformConfig = field(default_factory=MelSpecTransformConfig)
    feat_extractor: FeatExtractorConfig = field(default_factory=FeatExtractorConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    resample: ResampleConfig = field(default_factory=ResampleConfig)
    truncate: TruncateConfig = field(default_factory=TruncateConfig)
    compute_f0: ComputeF0Config = field(default_factory=ComputeF0Config)
    shuffle: ShuffleConfig = field(default_factory=ShuffleConfig)
    sort: SortConfig = field(default_factory=SortConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    padding: PaddingConfig = field(default_factory=PaddingConfig)
    train_conf: TrainConfig = field(default_factory=TrainConfig)
    train_conf_gan: TrainGanConfig = field(default_factory=TrainGanConfig)


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

if __name__ == '__main__':
    config = CosyVoice3Config()
    print(config)
