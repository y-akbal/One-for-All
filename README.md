# Hybrid-Wind-G
Some files related with the upcoming manuscript which revolves around ensemble learning for time series forecasting using transformers.
1) preprocessor.py is responsible for picking up the designated columns and normalizing them. Furthermore, if something goes wrong, a seperate log.txt file will be provided as well as the mean and standard deviations that are obtained as a result of normalization. To do so, you need to adjust the global variables inside the preprocessor.py file. 
2) layers.py contains some of the layers to be used. Most of the time we implement our own layers,
2.5) Attention layers contains multihead attention sort layers but a bit abstracted in some other sense.
3) data_pipe.py will contain necessary functions to grab the data out of csv files and do some preprocessing and prepare them.
