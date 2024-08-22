import os
import pandas as pd
import numpy as np


class memmap_array:
    def __init__(self, name="array.dat", directory = None):
        self.lengths = None ## Save for later use
        self.array = None ## Save for later use
        self.csv_names = None        
        self.name = name
        self.directory = directory
        
  
    def fit(self, dtype=np.float32):
        self.csv_names = [file for file in self.__getcsvlist__()]
        self.lengths = [
            len(pd.read_csv(file, low_memory=True)) for file in self.csv_names
        ]
        self.cum_len = np.cumsum([0] + self.lengths)
        self.len = sum(self.lengths)
        try:
            ## We save the concatted arrays
            self.array = np.memmap(self.name, shape=(self.len,), dtype=dtype, mode="w+")
            ## We save the lengths of the arrays for future use
            lengths = np.memmap(
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
        lengths[:] = self.lengths
        
        ## we now save the names of the csv files in a txt file -4 is for deleting .dat
        with open("names_" + self.name[:-4] + ".txt", mode = "w") as f:
            for name in self.csv_names:
                f.write(name + "\n")
        
        self.array.flush()  ## we write everything to the disk!!!!
        lengths.flush()  ## we write everything to the disk!!!!
        



    def __getcsvlist__(self):
        list_ = os.listdir(self.directory)
        csv = []
        for file in list_:
            if file.endswith(".csv"):
                csv.append(self.directory+"/"+file)
        return csv


    def to_memmap_array(self):
        """
        This dude will work in tandem with the memmmap arrays
        """
        return {"array": self.array, "lengths":self.lengths, "csv_names": self.csv_names}        

    def __preprocess__(self, frame):
        ### Here you can do whatever you like with the given frame object,
        ### In particular, things picking some columns, and getting rid of some large values

        return frame.iloc[:, -1]



if __name__ == "__main__":
    try:
        L = memmap_array()
        L.fit(dtype=np.float32)
        print(
        f"Memmap is initialized successfully. Total length is {len(L.array)}, and lengths are \n {L.lengths}. Thank you buddy!"
        )
    except Exception as e:
        print(e)
        
