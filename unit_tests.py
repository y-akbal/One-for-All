from abc import ABC, abstractmethod
import numpy as np
from torchsummary import summary

# add something on tensorboard


class model_test(ABC):  # This is called abstract base class
    """_summary_
    This class is designed to get intermediate information during the training process of a big network.
    The idea is to get shapes, to pass a fake data to see if the shapes are ok.
    if the model does at least one backward pass with fake data,
    and return some statistic with tensorflowboard, on time permitting.
    Since new model will be inherited from an abstract class, you can write your own files
    Here this dude will give a summary or something, for quick debugging purposes

    """

    @abstractmethod
    def __init__(self, model, input_dim, device="cuda"):
        self.model = model
        self.input_dim = input_dim
        self.device = device
        self.called = 0

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

    @abstractmethod
    def write_log(self, message: str):
        """_summary_
        The idea is to get intermediate reports and possible errors to see in a txt file
        Args:
            message (str): _description_
        returns none
        """
        num = self.called
        with open(self.file, mode="a") as file:
            file.write(f"{num}\t {message} \n")
        self.called += 1
