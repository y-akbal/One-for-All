
from abc import ABC, abstractmethod
import numpy as np


class model_test(ABC):  # This is called abstract base class
    @abstractmethod
    def __init__(self, model, input_dim):
        self.model = model

    @abstractmethod
    def test_shapes(self, model, input):
        pass

    @abstractmethod
    def list_output_shapes(self, x):
        return x**2
    
    @abstractmethod
    def count_params(self):
        lay = 0
        for i in self.model.parameters():
            lay += np.array(i.shape).prod()
        return lay
    
    
    