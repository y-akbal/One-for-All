import numpy as np


class ts_concatted:
    def __init__(self, array: np.ndarray, lengths: np.ndarray, lags: np.ndarray):
        """
        array should be concatted arrays,
        lengths should be the lengths of the individual arrays,
        lags should be the lags to be used,
        """
        assert (
            len(lags) > 1
        ), "Come on dude this is a time series, you gotta be a bit serious !!!!"
        assert len(array) == sum(
            lengths
        ), "For partition, sum of the lengths should be equal to total size of the time series concattenated"
        ## -- ##
        self.array = array
        self.lengths = np.cumsum(lengths)
        self.lags = lags
        self.horizons = [len_ - lag + 1 for len_, lag in zip(lengths, lags)]
        self.cumhors = np.cumsum(self.horizons)
        self.m = {i: j - i for i, j in enumerate(np.cumsum([0] + lags[:-1]))}

    def __place__(self, x: int, array: np.ndarray) -> int:
        return np.searchsorted(array, x, side="right")

    def __getitem__(self, i):
        if i > self.__len__() - 1:
            raise IndexError
        ### Otherwise go ahead my son ###
        place_ = self.__place__(i, self.cumhors)

        X = self.array[i + self.m[place_] : i + self.m[place_] + self.lags[place_]]
        N = len(X)
        return X[:-1], X[4:N:4], place_

    def __len__(self) -> int:
        return len(self.array) - sum(self.lags) + len(self.lags)

    def __read_csvfile__(self):
	pass


if __name__ == "__main__":
    print("Testing")
