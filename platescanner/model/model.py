from abc import ABC, abstractmethod
from cvtk.bbox import Bbox


class Model(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs) -> dict[str, Bbox]:
        raise NotImplementedError


__all__ = [
    'Model'
]
