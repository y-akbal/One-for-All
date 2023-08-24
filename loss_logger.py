import pickle
import os
import torch
from torch.distributed import (
    all_reduce,
    ReduceOp,
)


class distributed_loss_track:
    def __init__(self, project="time_series_pred", file_name: str = "loss.log"):
        self.project = project
        self.file_name = file_name
        self.temp_loss = 0
        self.counter = 1
        ## Bu kodu yazanlar ne güzel mühendislerdir, onların supervisorları ne
        ## iyi supervisorlardır

    def update(self, loss):
        self.temp_loss += loss
        self.counter += 1

    def get_avg_loss(self):
        return self.temp_loss / self.counter

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        loss_tensor = torch.tensor(
            [self.temp_loss, self.counter], device=device, dtype=torch.float32
        )
        all_reduce(loss_tensor, ReduceOp.SUM, async_op=False)
        self.temp_loss, self.counter = loss_tensor.tolist()


class loss_track:
    """
    Things will work as follows:

    Look at loss for an individual batch, update the weights
    update rolling loss, using self.update(loss) update rolling mean loss and
    variance, if you are done with final batch, call self.reset(),
    this will reset parameters, and flush the loss into a pickled file.
    This is important for illustration purposses.
    Pickle file is a dictionary file with the following form:

    {"mean_loss":self._temploss,
                 "var_loss":self._tempvar,
                 "loss_hist":self._loss_hist,
                 "loss_var_hist":self._lossvar_hist,
                 "num_epochs": self.num_epochs
                 }

    As we do not plan to pass a lot of epochs, we may be interested in variance and rolling mean,
    you can simply write a simple script to pick of mean loss, for each epoch.
    In the case that you need rolling loss to print, call self.loss. This will return a tuple
    mean and variance in the epoch.


    Turkish: Atanamamış W&B yaptık burada....
    """

    def __init__(self, project="time_series_pred", file_name: str = "loss.log") -> None:
        ## File name for logging loss logs ##
        self.file_name = file_name
        self.project_name = project
        ## --------------- ##
        ## We do not just keep track the mean loss
        ## but also standard deviation along the batches
        self._trainloss = 1e-10
        self._trainvar = 1e-10
        ##
        self._testloss = 1e-10
        self._testvar = 1e-10

        ## history »»
        self.train_loss_hist = [self._trainloss]
        self.train_loss_var_hist = [self._trainvar]

        self.test_loss_hist = [self._testloss]
        self.test_loss_var_hist = [self._testvar]

        ## counters for rolling mean and variance ##
        self.train_counter = 1
        self.test_counter = 1
        ### epochs
        self.num_epochs = 0

        if not os.path.isfile(file_name):
            try:
                self.pickled_file = file_name
            except Exception as e:
                print(e)

    def update_train(self, loss: float) -> None:
        ## update mean loss and variance loss ##
        difference = (self._trainloss - loss) / (self.train_counter + 1)
        self._trainloss -= difference  # to be used to update the variance
        self._trainvar = (
            ((self.train_counter - 1) / self.train_counter) * (self._trainvar)
            + ((loss - self._trainloss) ** 2) / (self.train_counter)
            + (difference) ** 2
        )
        ## append the real losses for future analysis
        self.train_loss_hist.append(self._trainloss)
        self.train_loss_var_hist.append(self._trainvar)
        ## update the counter
        self.train_counter += 1

    def update_test(self, loss: float) -> None:
        difference = (self._testloss - loss) / (self.test_counter + 1)
        self._testloss -= difference  # to be used to update the variance
        self._testvar = (
            ((self.test_counter - 1) / self.test_counter) * (self._testvar)
            + ((loss - self._testloss) ** 2) / (self.test_counter)
            + (difference) ** 2
        )
        ## append the real losses for future analysis
        self.test_loss_hist.append(self._testloss)
        self.test_loss_var_hist.append(self._testvar)
        ## update the counter
        self.test_counter += 1

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
            "num_epochs": self.num_epochs,
            "train_mean_loss": self._trainloss,
            "train_loss_variance": self._trainvar,
            "train_loss_hist": self.train_loss_hist,
            "train_loss_var_hist": self.train_loss_var_hist,
            "test_mean_loss": self._testloss,
            "test_loss_variance": self._testvar,
            "test_loss_hist": self.test_loss_hist,
            "test_loss_var_hist": self.test_loss_var_hist,
        }
        num = self.num_epochs
        with open(f"{num}_epoch" + self.file_name, mode="ab") as file:
            pickle.dump(dict_, file)

    @property
    def loss(self) -> tuple:
        return {
            "train_loss": self._trainloss,
            "train_variance": self._trainvar,
            "test_loss": self._testloss,
            "test_variance": self._testvar,
        }


if __name__ == "__main__":
    pass
