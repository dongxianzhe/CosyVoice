import torchaudio
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


def load_wav(wav: str, target_sr: int, min_sr: int=16000):
    speech, sample_rate = torchaudio.load(wav, backend='soundfile') # (channels, frames)
    speech = speech.mean(dim=0, keepdim=True) # (channels=1, frames)
    if sample_rate != target_sr:
        assert sample_rate >= min_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech) # (1, frames / sample_rate * target_sr)
    return speech