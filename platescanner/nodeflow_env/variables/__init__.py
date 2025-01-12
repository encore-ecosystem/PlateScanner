from nodeflow import Variable
from pathlib import Path


class MyPath(Variable):
    def __init__(self, value: Path):
        super().__init__(value)

    def __truediv__(self, other: str) -> Path:
        return self.value / other


__all__ = ["MyPath"]
