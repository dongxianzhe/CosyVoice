import os
from typing import Iterator, Any
import torch
from safetensors import safe_open


def iter_model_files(folder_path: str) -> Iterator[str]:
    """递归遍历目录，返回所有 .safetensors 和 .pt 文件路径"""
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.endswith((".safetensors", ".pt")):
                yield os.path.join(root, fname)


def inspect_safetensors_file(path: str) -> None:
    print(f"\n=== File (safetensors): {path} ===")

    cnt = 0
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor: torch.Tensor = f.get_tensor(key)
            print(
                f"{key}: "
                f"shape={tuple(tensor.shape)}, "
                f"dtype={tensor.dtype}"
            )
            cnt += tensor.numel()
    print(f"\n=== File (safetensors): {path} total params {cnt} ===")


def inspect_pt_file(path: str) -> None:
    print(f"\n=== File (pt): {path} ===")

    obj: Any = torch.load(path, map_location="cpu")

    cnt = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if torch.is_tensor(v):
                print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
                cnt += v.numel()
            elif isinstance(v, dict):
                print(f"{k}: <dict>")
            else:
                print(f"{k}: {type(v)}")
    else:
        print(f"Loaded object type: {type(obj)}")
    print(f"\n=== File (safetensors): {path} total params {cnt} ===")


def inspect_models(folder_path: str) -> None:
    for file_path in iter_model_files(folder_path):
        if file_path.endswith(".safetensors"):
            inspect_safetensors_file(file_path)
        elif file_path.endswith(".pt"):
            inspect_pt_file(file_path)


if __name__ == "__main__":
    folder: str = "/data/home/xianzhedong/models/Fun-CosyVoice3-0.5B"
    inspect_models(folder)