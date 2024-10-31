from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError


__all__ = [
    'Model'
]
