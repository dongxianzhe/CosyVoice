from os.path import expanduser
from huggingface_hub import snapshot_download

MODELS = [
    ('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', 'Fun-CosyVoice3-0.5B'),
    ('FunAudioLLM/CosyVoice-ttsfrd', 'CosyVoice-ttsfrd'),
]
BASE_DIR = expanduser('~/models')

for repo_id, local_name in MODELS:
    local_dir = f'{BASE_DIR}/{local_name}'
    print(f'Downloading {repo_id} -> {local_dir} ...')
    snapshot_download(repo_id, local_dir=local_dir)
    print(f'Done: {local_dir}')

print('All models downloaded.')
