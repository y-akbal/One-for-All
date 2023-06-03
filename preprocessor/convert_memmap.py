import os
import pandas as pd
import numpy as np


class memmap_array:
    def __init__(self, name="array.dat"):
        self.lengths = None
        self.array = None
        self.name = name
        self.fitted = False

    def __getitem__(self, i):
        assert self.fitted, "You should first fit your array"
        if i < self.len:
            return self.array[i]
        else:
            raise IndexError

    def fit(self, dtype=np.float32):
        self.lengths = [
            len(pd.read_csv(file, low_memory=True)) for file in self.__getcsvlist__()
        ]
        self.cum_len = np.cumsum([0] + self.lengths)
        self.len = sum(self.lengths)
        try:
            ## We save the concatted arrays
            self.array = np.memmap(self.name, shape=(self.len,), dtype=dtype, mode="w+")
            ## We save the lengths of the arrays for future use
            Lengths = np.memmap(
                "lengths" + self.name,
                shape=(len(self.lengths),),
                dtype=np.uint32,
                mode="w+",
            )
        except Exception as exception:
            raise exception

        ## below we place the concatted arrays ---
        for index, file in enumerate(self.__getcsvlist__()):
            l_index = self.cum_len[index]
            h_index = self.cum_len[index + 1]

            csv = pd.read_csv(file)
            csv = self.__preprocess__(csv)

            self.array[l_index:h_index] = csv.to_numpy()

        ## lengths are saved here ##
        Lengths[:] = self.lengths

        self.array.flush()  ## we write everything to the disk!!!!
        Lengths.flush()  ## we write everything to the disk!!!!

        self.fitted = True  ## set fitted to true

    def convert_nparray(self):
        self.array = np.array(self.array)

    def __getcsvlist__(self):
        list_ = os.listdir()
        csv = []
        for file in list_:
            if file.endswith(".csv"):
                csv.append(file)
        return csv

    def __len__(self):
        return self.len

    @classmethod
    def from_file(cls, name="array.dat"):
        cls_ = cls()
        cls_.array = np.memmap(name, mode="r", dtype=np.float32)
        cls_.name = name
        cls_.lengths = np.memmap("lengths" + name, mode="r", dtype=np.uint32)
        cls_.len = len(cls_.array)
        cls_.fitted = True
        return cls_

    def __preprocess__(self, frame):
        return frame.iloc[:, -1]


if __name__ == "__main__":
    L = memmap_array()
    L.fit(dtype=np.float32)
