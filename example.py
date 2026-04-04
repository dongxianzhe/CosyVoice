import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio


def cosyvoice3_example():
    """ CosyVoice3 Usage, check https://funaudiollm.github.io/cosyvoice3/ for more details
    """
    cosyvoice = AutoModel(model_dir='/data/home/xianzhedong/models/Fun-CosyVoice3-0.5B')
    for i, j in enumerate(cosyvoice.inference_zero_shot('今天天气真好，能陪我出去逛逛吗？', 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                                                        './asset/zero_shot_prompt.wav', stream=False)):
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

def main():
    cosyvoice3_example()

if __name__ == '__main__':
    main()
