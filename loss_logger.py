import pickle
import os


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
        self._temploss = 1e-10
        self._tempvar = 1e-10
        ##
        self._loss_hist = [self._temploss]
        self._lossvar_hist = [self._tempvar]
        ## counters for rolling mean and variance ##
        self.counter = 1
        self.num_epochs = 0

        if not os.path.isfile(file_name):
            try:
                self.pickled_file = file_name
            except Exception as e:
                print(e)

    def update(self, loss: float) -> None:
        ## update mean loss and variance loss ##
        difference = (self._temploss - loss) / (self.counter + 1)
        self._temploss -= difference  # to be used to update the variance
        self._tempvar = (
            ((self.counter - 1) / self.counter) * (self._tempvar)
            + ((loss - self._temploss) ** 2) / (self.counter)
            + (difference) ** 2
        )
        ## append the real losses for future analysis
        self._loss_hist.append(self._temploss)
        self._lossvar_hist.append(self._tempvar)
        ## update the counter
        self.counter += 1

    def reset(self) -> None:
        self.num_epochs += 1
        self.__flush__()
        ### we reset all variables ###
        self._temploss = 1e-10
        self._tempvar = 1e-10
        self.counter = 1
        ### update the number of epochs passed

    def __flush__(self) -> None:
        dict_ = {
            "project_name": self.project_name,
            "mean_loss": self._temploss,
            "var_loss": self._tempvar,
            "loss_hist": self._loss_hist,
            "loss_var_hist": self._lossvar_hist,
            "num_epochs": self.num_epochs,
        }
        num = self.num_epochs
        with open(f"{num}_epoch" + self.file_name, mode="ab") as file:
            pickle.dump(dict_, file)

    @property
    def loss(self) -> tuple:
        return self._temploss, self._tempvar

    @loss.getter
    def loss(self) -> tuple:
        return self._temploss, self._tempvar


if __name__ == "__main__":
    pass
