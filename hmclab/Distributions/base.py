from abc import ABC, abstractmethod


class AbstractDistribution(ABC):
    dimensionality: int = 1
    name: str = ""

    def __init__(self):
        pass  # pragma: no cover

    @abstractmethod
    def log_prob(self, x):
        pass  # pragma: no cover

    @abstractmethod
    def log_prob_grad(self, x):
        pass  # pragma: no cover
