# One for All: A Method for forecasting multiple time series in a univariate sense using a single model.
Some files related to the upcoming manuscript which revolves around ensemble learning for time series forecasting using transformers.
1) preprocessor.py is responsible for picking up the designated columns and normalizing them. Furthermore, if something goes wrong, a seperate log.txt file will be provided as well as some statistical properties that are obtained as a result of normalization. To do so, you need to adjust the global variables inside the preprocessor.py file. 
2) layers.py contains some of the layers to be used. For the sake of completeness we implement our own layers,
3) preprocess.py will contain necessary functions to grab the data out of csv files and do some preprocessing and prepare them,
4) convert_memmap.py wll convert will grab csv files convert them into data.dat and lengths.dat files.
5) memmap_arrays.py is a signature snippet, grabbing data.dat and lengths.dat files, and will give you a window of given size from your concatted time series, as X[:-1], X[-1] and place_, where place_ indicates which time series to be used.
6) RF.py is used to do old-school machine learning analysis -  given that all csv.files are preprocessed - you can choose for instance randomforest or KNN sort of old school stuff. Splitting of time series is done on the fly. Multiprocessing is supported. At the end of the procedure, a file will be provided to see the results.


Things to do:
  5) Unit_tests.py will contain a base class that contain some utility and unit tests to test the model. 
  6) train_details.py - probably training such a huge model will take a lot of time, therefore we may need to parallelize training process. This file will contain some functions to achieve this.
  
  



