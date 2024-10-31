from ..datasets import AbstractDataset
from ..adapters import AbstractAdapter
from ..adapters.enum import AdapterOutputType


class Combiner:
    @classmethod
    def merge(cls, adapters: list[AbstractAdapter], output_type: AdapterOutputType) -> AbstractDataset:
        result_dataset = adapters[0].forward(output_type)
        for adapter in adapters[1:]:
            result_dataset += adapter.forward(output_type)
        return result_dataset

__all__ = [
    'Combiner'
]
