import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic
from preprocessor.convert_memmap import memmap_array

import warnings
warnings.filterwarnings("ignore")


### Some
NEW_PATH = "preprocessed_data"
COLUMN_NAMES = ["Tarih", "Saat", "Toplam (MWh)"]
SPLIT_RATIO = 0.85 # Train Test split ratio


class logger:
    def __init__(self, file="log.txt") -> None:
        self.called = 0
        self.file = file

    def write_log(self, message: str):
        num = self.called
        with open(self.file, mode="a") as file:
            file.write(f"{num}\t {message} \n")
        self.called += 1


class CreationError(Exception):
    def __init__(
        self,
        message=" You should make sure that the target directory has not been created before!",
    ):
        self.message = message
        super().__init__(self.message)


def get_csv_list(excluded_files:list = ["results.csv","results_scores.csv"]) -> list:
    list_ = os.listdir()
    csv = []
    for file in list_:
        if file.endswith(".csv") and file not in excluded_files:
            csv.append(file)
    return csv


def statistical_results(ts: pd.Series, advanced = False) -> dict:
    ### here you write all statistical properties of your time series
    ##below we collect necessary statistics
    if advanced:
        ord = arma_order_select_ic(ts[:2000], max_ar=10, max_ma=2, ic=["aic", "bic"])
        statistic_dict = {
            "Mean": ts.mean(),
            "Std": ts.std(ddof=1),
            "Max": ts.max(),
            "Min": ts.min(),
            "length": ts.shape[0],
            "p_value": adfuller(ts)[1],
            "aic_order": ord.aic_min_order,
            "bic_order": ord.bic_min_order,
        }
    else:
        statistic_dict = {
            "Mean": ts.mean(),
            "Std": ts.std(ddof=1),
            "Max": ts.max(),
            "Min": ts.min(),
            "length": ts.shape[0],
        }

    return statistic_dict


def preprocess_csv(csv_file: str, numerical_column:int=-1) -> tuple:
    pandas_frame = pd.read_csv(csv_file, encoding="ISO-8859-1", decimal=",")
    try:
        pandas_frame = pandas_frame.loc[:, COLUMN_NAMES]
    except Exception:
        print(f"The file {csv_file} is probably corrupted!!!")
        raise(Exception)

    numerical_series = pandas_frame.iloc[:, numerical_column].copy(deep=True)
    ### --- Some Preprocessing ---- ###
    ### --- We eliminate negative values --- ### 
    numerical_series[numerical_series < 0] = np.nan
    numerical_series = numerical_series.interpolate(method="linear")
    pandas_frame.iloc[:, numerical_column] = numerical_series

    ### Normalizing stuff
    time_series = pandas_frame.iloc[:, numerical_column]  ### cleared time series
    #### Here we need to have p-values of unit-root tests, PACF results ACF results to get results on
    statistics:dict = statistical_results(time_series)
    
    mean = statistics["Mean"]
    std = statistics["Std"]
    # max = statistics["Max"]
    # min = statistics["Min"]
    #### exclude columns ## ---
    pandas_frame.iloc[:, numerical_column] = (
        pandas_frame.iloc[:, numerical_column] - mean
    ) / std
    return (
        pandas_frame,
        statistics,
    )  ### this guy is the normalized frame BTW!


def main() -> None:
    ############# Create the main directory to copy preprocessed files #################
    try:
        if SPLIT_RATIO > 0:
            os.mkdir(NEW_PATH+"_test")
            os.mkdir(NEW_PATH+"_train")
        os.mkdir(NEW_PATH)
    except FileExistsError:
        raise (CreationError)
    logger_ = logger()

    ## Time to run ##
    list_csv = get_csv_list()
    assert len(list_csv) > 0, "There should be at least one CSV file!"  ### assertion!
    iterator_csv = tqdm(list_csv)  ## get the iterator of csv files
    ## ---------------------------------------------------------------- ###
    message = 0
    spatial_statistics = dict()
    for csv_file in iterator_csv:
        ### check if something goes wrong!
        try:
            preprocessed_pd, statistics = preprocess_csv(csv_file)
            #name = os.path.join(NEW_PATH, csv_file)
            #preprocessed_pd.to_csv(name)
            N_ = len(preprocessed_pd)
            if SPLIT_RATIO > 0:
                ## If you really like to split the dataset
                N = int(N_*SPLIT_RATIO)
                for T  in ["_train", "_test", ""]:

                    if T == "_train":
                        name_train = os.path.join(NEW_PATH+T, csv_file)
                        preprocessed_pd.iloc[:N,:].to_csv(name_train)
                    elif T == "_test":
                        name_test = os.path.join(NEW_PATH+T, csv_file)
                        preprocessed_pd.iloc[N:,:].to_csv(name_test)
                    else:
                        name_main = os.path.join(NEW_PATH+T, csv_file)
                        preprocessed_pd.to_csv(name_main)
                ### Now record the statistics to a dataframe ###                    
                
                ##watchaa the order this is important
            else:
                name_main = os.path.join(NEW_PATH, csv_file)
                preprocessed_pd.to_csv(name_main)
            spatial_statistics[csv_file] = statistics
        except Exception as exception:
            message += 1
            logger_.write_log(f"{exception} is wrong with {csv_file}")
    ### Now convert statistics to pandas data frame ---
    
    data_frame = pd.DataFrame().from_dict(spatial_statistics)
    data_frame = data_frame.transpose()
    data_frame.to_csv("results.csv")

    ### Final Stuff ###
    if message > 0:
        print(
            f"Something went wrong with {message} csv files, see the log.txt for a detailed description!\n"
        )
    else:
        print("All csv files were processed successfully!")
        print("See also results.csv for the means and the standard deviations.")
    return data_frame


if __name__ == "__main__":
    parent_dir = os.getcwd()
    try:
        df = main()
        print("Preprocessing is done!")
        for T in ["_train", "_test"]:
            directory = NEW_PATH+T
            T_ = memmap_array("array"+f"{T}.dat", directory = directory)
            T_.fit(dtype = np.float32)
    except Exception as exception:
        print(f"Something went wrong!!! Here is the thing {exception}")
        
