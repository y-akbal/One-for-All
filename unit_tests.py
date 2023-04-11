
from abc import ABC, abstractmethod
import numpy as np
from torchsummary import summary


class model_test(ABC):  # This is called abstract base class
    @abstractmethod
    def __init__(self, model, input_dim, device = "cuda"):
        self.model = model
        self.input_dim = input_dim
        self.device = device

    @abstractmethod
    def test_shapes(self, model, input):
        pass

    @abstractmethod
    def list_output_shapes(self, x):
        return x**2
    
    @abstractmethod
    def count_params(self):
        L = []
        lay = 0
        for i in self.model.parameters():
            lay += np.array(i.shape).prod()
            L.append(i.device)
        return lay
    
    
    