import torch
from torch import nn as nn
from torch.nn import functional as F
from layers import block, Upsampling, Linear
import pickle

class Model(nn.Module):
    def __init__(
        self,
        lags: int = 512,
        embedding_dim: int = 512,
        n_blocks: int = 25,
        pool_size: int = 4,
        number_of_heads=8,
        number_ts=25,
        num_of_clusters=None,  ### number of clusters of times series
        channel_shuffle_group=2,  ## active only and only when channel_shuffle is True
    ):
        assert (
            lags / pool_size
        ).is_integer(), "Lag size should be divisible by pool_size"
        super().__init__()
        self.width = lags // pool_size
        self.embedding_dim = embedding_dim
        ###
        self.cluster_used = True if num_of_clusters is not None else False
        ###
        self.blocks = nn.Sequential(
            *(
                block(
                    embedding_dim,
                    width=self.width,
                    n_heads=number_of_heads,
                )
                for _ in range(n_blocks)
            )
        )
        self.up_sampling = Upsampling(
            lags=lags,
            d_out=self.embedding_dim,
            pool_size=pool_size,
            num_of_ts=number_ts,
            conv_activation=F.gelu,
            num_of_clusters=num_of_clusters,
            channel_shuffle_group=channel_shuffle_group,
        )

        ###
        self.Linear = Linear(self.embedding_dim, 1)
        ###
        ### here is the config dict to be used
        self.config = {"lags":lags, 
                       "embedding_dim":embedding_dim,
                       "n_blocks":n_blocks,
                       "pool_size":pool_size,
                       "number_of_heads":number_of_heads, 
                       "number_ts":number_ts,
                       "num_of_clusters":num_of_clusters,
                       "channel_shuffle_group":channel_shuffle_group
                       }
        

    @classmethod 
    def from_config_file(cls, config_file):
        with open(config_file, mode = "rb") as file:
            dict_ = pickle.load(file)
        if isinstance(dict_, dict):
            return cls(**dict_)
        else:
            raise ValueError("The pickle file should contain a config dictionary")
    
    ### Let's these dudes stay here for future versions ###        
    @classmethod
    def from_pretrained(cls, file_name):
        try:
            dict_ = torch.load(file_name)
            config = dict_["config"]
            state_dict = dict_["state_dict"]
            model = cls(**config)    
            model.load_state_dict(state_dict)
            print("Model loaded successfully!!!!")
        except Exception as e:
            print(f"Something went wrong with {e}")        
        return model
      
    @classmethod
    def from_data_class(cls, data_class):
        if isinstance(data_class, dict):
            return cls(**data_class)
        else:
            return cls(**data_class.__dict__)
    
    def write_config_file(self, file_name):
        with open(file_name, mode = "wb") as file:
            pickle.dump(self.config,file)
    
    def save_model(self, file_name):
        fn = "Model" if file_name == None else file_name
        model = {}
        model["state_dict"] = self.state_dict()
        model["config"] = self.config
        try:
            torch.save(model, f"{fn}")
            print(f"Model saved succesfully, see the file {fn} for the weights and config file!!!")
        except Exception as exp:
            print(f"Something went wrong with {exp}!!!!!")
         
         
        
    def forward(self, x):
        ## Here we go with upsampling layer
        if self.cluster_used:
            x, tse_embedding, cluster_embedding = x[0], x[1], x[2]
            x = self.up_sampling((x, tse_embedding, cluster_embedding))
        else:
            x, tse_embedding = x[0], x[1]
            x = self.up_sampling((x, tse_embedding))
        ## Concatted transformer blocks
        ###
        x = self.blocks(x)
        return self.Linear(x)


"""
model = Model()

model = Model.from_pretrained("10epoch.pt")
model = Model.from_config_file("write_it.cfg")

torch.manual_seed(0)
model([torch.randn(1, 1, 512), torch.tensor([0])])

model.save_model("10epoch.r")
        
"""


