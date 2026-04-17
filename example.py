import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice3
import torchaudio
from datetime import datetime

CN_DIGITS = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
             '十一', '十二', '十三', '十四', '十五', '十六', '十七', '十八', '十九', '二十',
             '二十一', '二十二', '二十三', '二十四', '二十五', '二十六', '二十七', '二十八', '二十九', '三十',
             '三十一', '三十二', '三十三', '三十四', '三十五', '三十六', '三十七', '三十八', '三十九', '四十',
             '四十一', '四十二', '四十三', '四十四', '四十五', '四十六', '四十七', '四十八', '四十九', '五十',
             '五十一', '五十二', '五十三', '五十四', '五十五', '五十六', '五十七', '五十八', '五十九']

def get_time_text() -> str:
    now = datetime.now()
    return f'现在的时间是{CN_DIGITS[now.hour]}点{CN_DIGITS[now.minute]}分。'


def cosyvoice3_example():
    cosyvoice = CosyVoice3(model_dir='/data/home/xianzhedong/models/Fun-CosyVoice3-0.5B')
    tts_text = get_time_text()
    print(f'TTS text: {tts_text}')
    for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text, prompt_text='You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。', prompt_wav='./asset/zero_shot_prompt.wav', stream=False)):
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

if __name__ == '__main__':
    cosyvoice3_example()
