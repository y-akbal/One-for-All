import pandas as pd
import numpy as np
import os


def return_csv_files(n_number:int = 500)-> None:
    lengths = []
    for i in range(n_number):
        length = np.random.randint(200, 500)
        lengths.append(length)
        array = i*np.ones(length, dtype = np.float32)
        pd.DataFrame(array).to_csv(f"{i}.csv")
    pd.DataFrame(lengths).to_csv("lengths.csv")
    return None


if __name__ == "__main__":
    try:
        return_csv_files()
    except Exception as e:
        print(f"Something went wrong with {e}")
    
    
