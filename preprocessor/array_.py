from convert_memmap import memmap_array

from  memmap_arrays import ts_concatted
import numpy as np


array = np.memmap("array.dat", mode = "r", dtype = np.float32)
lengths = np.memmap("lengthsarray.dat", mode = "r", dtype = np.uint32)



