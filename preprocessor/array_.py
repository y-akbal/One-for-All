from convert_memmap import memmap_array
from  memmap_arrays import ts_concatted
import numpy as np
from torch.utils.data import DataLoader


array = np.memmap("array.dat", mode = "r", dtype = np.float32)
lengths = np.memmap("lengthsarray.dat", mode = "r", dtype = np.uint16)

data = ts_concatted(array, lengths,lags = [23 for _ in lengths])


data_ = DataLoader(data, batch_size = 16, shuffle = True)
import time

a = time.time()
for x, y, z in data_:
    x.to("cuda")
    y.to("cuda")    
    z.to("cuda")
    pass

l = time.time() -a 

