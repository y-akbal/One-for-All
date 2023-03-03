import os
import pandas as pd
import numpy as np
from tqdm import tqdm

### Some 
NEW_PATH = "preprocessed_data"
COLUMN_NAMES = ["Tarih", "Saat", "Toplam (MWh)"]

class logger:
    def __init__(self, file = "log.txt") -> None:
        self.called = 0
        self.file = file
    def write_log(self, message: str):
        num = self.called
        with open(self.file, mode = "a") as file:
            file.write(f"{num}\t {message} \n")
        self.called += 1
        

def get_csv_list() -> list:
    list_ = os.listdir()
    csv = []
    for file in list_:
        if file.endswith(".csv"):
            csv.append(file)
    return csv



def preprocess_csv(csv_file : str, numerical_column = -1) -> pd.DataFrame:
    pandas_frame = pd.read_csv(csv_file, encoding = "ISO-8859-1", decimal = ',')
    pandas_frame = pandas_frame.loc[:, COLUMN_NAMES]
    if pandas_frame.isnull().values.any():
        pass
    pandas_frame.iloc[:, numerical_column] = pd.Series.interpolate(pandas_frame.iloc[:, numerical_column])
    return pandas_frame


def main() -> None:
    ############# Create the main directory to copy preprocessed files #################
    try:
        os.mkdir(NEW_PATH)
    except FileExistsError:
        print("The directory already exists!")
    logger_ = logger()    
        
    ## Time to run ##  
    list_csv = get_csv_list()
    assert len(list_csv) > 0, "There should be at least one CSV file!" ### assertion!
    iterator_csv = tqdm(list_csv) 
    #
    message = 0
    for csv_file in iterator_csv:
        ### check if something goes wrong!
        try: 
            preprocessed_pd = preprocess_csv(csv_file)
            name = os.path.join(NEW_PATH, csv_file)
            preprocessed_pd.to_csv(name) 

        except Exception: 
            
            message += 1
            logger_.write_log(f"{Exception} is wrong with {csv_file}")
    
    ### Final Stuff ###
    if message > 0:
        print(f"Something went wrong with {message} csv files, see the log.txt for detailed description!")
    else:
        print("All csv files were processed successfully!")
    return None 


if __name__ == "__main__":
    main()




