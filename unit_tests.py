
from abc import ABC, abstractmethod
import numpy as np


class model_test(ABC):  # This is called abstract base class
    @abstractmethod
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def test_shapes(self, model, input):
        pass

    @abstractmethod
    def output_shapes(self, x):
        return x**2

    @abstractmethod
    def count_params(self):
        lay = 0
        for i in self.layer.parameters():
            lay += np.array(i.shape).prod()
        return lay
    