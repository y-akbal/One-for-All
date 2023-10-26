import pickle
import os
import torch
from torch.distributed import (
    all_reduce,
    ReduceOp,
)

class loss_track:
    def __init__(self, project="time_series_pred", file_name: str = "loss.log"):
        self.project = project
        self.file_name = file_name
        self.temp_loss = 0
        self.counter = 1
        self.loss = []

    def update(self, loss):
        self.temp_loss += loss
        self.counter += 1

    def reset(self):
        self.temp_loss = 0
        self.counter = 1

    def get_loss(self):
        if self.counter == 0:
            return self.temp_loss
        return self.temp_loss / self.counter

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            loss_tensor = torch.tensor(
            [self.temp_loss, self.counter], device=device, dtype=torch.float32
            )
            all_reduce(loss_tensor, ReduceOp.SUM, async_op=True)
            self.temp_loss, self.counter = loss_tensor.tolist()
            self.loss.append(self.temp_loss)
        else:
            self.loss.append(self.temp_loss)


class loss_track:
    """
    This dudes keeps track of the history of the loss in a rolling mean sense!!!
    """
    def __init__(self, project="time_series_pred", 
                 file_name: str = "loss.log", 
                 initial_val_for_loss = 1e-10,
                 initial_val_for_variance = 1,
                 ) -> None:
        ## File name for logging loss logs ##
        self.file_name = file_name
        self.project_name = project
        ## --------------- ##
        ## We do not just keep track the mean loss
        ## but also standard deviation along the batches
        self._loss = initial_val_for_loss
        self._var = initial_val_for_variance
        ##
        ## history »»
        self._loss_hist = [self._loss]
        self._var_hist = [self._var]

        ## counters for rolling mean and variance ##
        self.counter = 1
        ### epochs
        self.num_epochs = 0

        if not os.path.isfile(file_name):
            try:
                self.pickled_file = file_name
            except Exception as e:
                print("We start from zero hope to go to zero in a very short period!!!")

    def update(self, loss: float) -> None:
        ## update mean loss and variance loss ##
        difference = (self._loss - loss) / (self.counter + 1)
        self._loss -= difference  # to be used to update the variance
        self._var = (
            ((self.counter - 1) / self.counter) * (self._var)
            + ((loss - self._loss) ** 2) / (self.counter)
            + (difference) ** 2
        )
        ## append the real losses for future analysis
        self._loss_hist.append(self._loss)
        self._var_hist.append(self._var)
        ## update the counter
        self.counter += 1

    def reset(self) -> None:
        ### update the number of epochs passed
        self.num_epochs += 1
        self.__flush__()
        ## -- ##
        self.train_counter = 1
        self.test_counter = 1

    def __flush__(self) -> None:
        dict_ = {
            "project_name": self.project_name,
            "Epoch": self.num_epochs,
            "Loss": self._loss,
            "Variance": self._var,
            "Loss History": self._loss_hist,
            "Variance History": self._var_hist            
        }
        num = self.num_epochs
        with open(f"epoch_{num}" + self.file_name, mode="ab") as file:
            pickle.dump(dict_, file)

    @property
    def loss(self) -> tuple:
        return {
            "loss": self._loss,
            "variance": self._var,
            }

"""
loser = loss_track()
loser.update(12)
loser.loss
"""

if __name__ == "__main__":
    print("OK Boomer!!!")
