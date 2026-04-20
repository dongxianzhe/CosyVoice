import time


class Timer:
    def __init__(self, name: str = ""):
        self.name: str = name
        self.start: float = 0.
        self.cost: float = 0.

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.cost = time.time() - self.start
        print(f"[{self.name}] cost {self.cost:.4f}s")
