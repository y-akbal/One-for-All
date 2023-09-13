import numpy as np


class ts_concatted:
    def __init__(self, array: np.ndarray, lengths: np.ndarray, lags: np.ndarray, file_names:str = None):
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
        self.file_names = file_names
        self.horizons = [len_ - lag + 1 for len_, lag in zip(lengths, lags)]
        self.cumhors = np.cumsum(self.horizons)
        self.m = {i: j - i for i, j in enumerate(np.cumsum([0] + lags[:-1]))}
        ## -- given that you provide txt files -- ##
        if file_names is not None:
            self.__read_csvfile_names__()

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
    ### Addendum to 
    def __read_csvfile_names__(self):
        with open(self.file_names, mode = "r") as file:
            file_names = file.readlines()
        file_names_mapped = list(map(self.__preprocess_file_names__, file_names))
        self.__file_names__ = {i: file_names_mapped[i] for i, _ in enumerate(self.lengths)}
        
    
    def __preprocess_file_names__(self, file_name:str) -> str:
        return file_name

    def return_file_names(self, place:int) -> str:
        return self.__file_names__[place]
        
"""
lags =  [512 for _ in np.memmap("lengthsarray_train.dat", dtype = np.uint32)]
lengths = np.memmap("lengthsarray_train.dat", dtype = np.uint32)
array = np.memmap("array_train.dat", dtype = np.float32)
L = ts_concatted(array = array, lengths = lengths,lags = lags, file_names="namesarray_train.txt")

L[0][-1]

L.return_file_names(L[1565111][-1])
"""
if __name__ == "__main__":
    print("Testing")
