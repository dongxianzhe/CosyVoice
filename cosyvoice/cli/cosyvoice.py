import os
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoice3Model
from cosyvoice.utils.file_utils import logging


class CosyVoice3:
    def __init__(self, model_dir: str):
        hyper_yaml_path = f'{model_dir}/cosyvoice3.yaml'.format()
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        # get_tokenizer: !name:cosyvoice.tokenizer.tokenizer.get_qwen_tokenizer
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

    def inference_zero_shot(self, tts_text, prompt_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_wav, self.sample_rate, zero_shot_spk_id)
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                yield model_output