import os
from functools import partial
from tqdm import tqdm
from omegaconf import DictConfig
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoice3Model, TTSInputParams
from cosyvoice.utils.file_utils import logging
from cosyvoice.llm.llm import CosyVoice3LM, Qwen2Encoder
from cosyvoice.utils.common import ras_sampling
from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer
from cosyvoice.flow.flow_matching import CausalConditionalCFM
from cosyvoice.flow.DiT.dit import DiT
from cosyvoice.hifigan.generator import CausalHiFTGenerator
from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
from matcha.utils.audio import mel_spectrogram


def get_config(model_dir: str) -> dict:
    qwen_pretrain_path = os.path.join(model_dir, 'CosyVoice-BlankEN')

    # fixed params
    sample_rate = 24000
    llm_input_size = 896
    llm_output_size = 896
    spk_embed_dim = 192
    token_mel_ratio = 2
    chunk_size = 25
    num_decoding_left_chunks = -1

    return {
        'sample_rate': sample_rate,
        'allowed_special': 'all',
        'get_tokenizer': partial(get_qwen_tokenizer,
                                 token_path=qwen_pretrain_path,
                                 skip_special_tokens=True,
                                 version='cosyvoice3'),
        'feat_extractor': partial(mel_spectrogram,
                                  n_fft=1920, num_mels=80,
                                  sampling_rate=sample_rate,
                                  hop_size=480, win_size=1920,
                                  fmin=0, fmax=None, center=False),
        'llm': CosyVoice3LM(
            llm_input_size=llm_input_size,
            llm_output_size=llm_output_size,
            speech_token_size=6561,
            length_normalized_loss=True,
            lsm_weight=0,
            mix_ratio=[5, 15],
            llm=Qwen2Encoder(pretrain_path=qwen_pretrain_path),
            sampling=partial(ras_sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1),
        ),
        'flow': CausalMaskedDiffWithDiT(
            input_size=80,
            output_size=80,
            spk_embed_dim=spk_embed_dim,
            output_type='mel',
            vocab_size=6561,
            input_frame_rate=25,
            only_mask_loss=True,
            token_mel_ratio=token_mel_ratio,
            pre_lookahead_len=3,
            pre_lookahead_layer=PreLookaheadLayer(
                in_channels=80, channels=1024, pre_lookahead_len=3,
            ),
            decoder=CausalConditionalCFM(
                in_channels=240,
                n_spks=1,
                spk_emb_dim=80,
                cfm_params=DictConfig(content={
                    'sigma_min': 1e-06,
                    'solver': 'euler',
                    't_scheduler': 'cosine',
                    'training_cfg_rate': 0.2,
                    'inference_cfg_rate': 0.7,
                    'reg_loss_type': 'l1',
                }),
                estimator=DiT(
                    dim=1024, depth=22, heads=16, dim_head=64,
                    ff_mult=2, mel_dim=80, mu_dim=80, spk_dim=80,
                    out_channels=80,
                    static_chunk_size=chunk_size * token_mel_ratio,
                    num_decoding_left_chunks=num_decoding_left_chunks,
                ),
            ),
        ),
        'hift': CausalHiFTGenerator(
            in_channels=80,
            base_channels=512,
            nb_harmonics=8,
            sampling_rate=sample_rate,
            nsf_alpha=0.1,
            nsf_sigma=0.003,
            nsf_voiced_threshold=10,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            istft_params={'n_fft': 16, 'hop_len': 4},
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            lrelu_slope=0.1,
            audio_limit=0.99,
            conv_pre_look_right=4,
            f0_predictor=CausalConvRNNF0Predictor(
                num_class=1, in_channels=80, cond_channels=512,
            ),
        ),
    }


class CosyVoice3:
    def __init__(self, model_dir: str):
        configs = get_config(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          f'{model_dir}/campplus.onnx',
                                          f'{model_dir}/speech_tokenizer_v3.onnx',
                                          f'{model_dir}/spk2info.pt',
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        self.model = CosyVoice3Model(configs['llm'], configs['flow'], configs['hift'])
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        del configs

    def inference_zero_shot(self, tts_text: str, prompt_text: str, prompt_wav: str, zero_shot_spk_id: str='', stream: bool=False, speed: float=1.0, text_frontend=True):
        prompt_text: str = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        print(f'prompt_text {prompt_text}')
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            print(f'tts_text {tts_text}')
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_wav, self.sample_rate, zero_shot_spk_id)
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(TTSInputParams(**model_input, stream=stream, speed=speed)):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                yield model_output

