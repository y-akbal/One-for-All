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
        return cls(**dict_)
    @classmethod
    def from_pretrained(cls, file_name, config_file):
        non_trained_model = cls.from_config_file(config_file)
        non_trained_model.load_state_dict(torch.load(file_name))
        return non_trained_model
        
    @classmethod
    def from_data_class(cls, data_class):
        return cls(**data_class.__dict__)
    
    def write_config_file(self, file_name):
        with open(file_name, mode = "wb") as file:
            pickle.dump(self.config,file)
    
    def save_model(self, file_name = None):
        fn = "Model" if file_name == None else file_name
        try:
            torch.save(self.state_dict(), f"{fn}.trc")
            self.write_config_file("config_file"+fn + ".cfg")
            print("Model saved succesfully, see {fn}.trc files  for the weight")
        except Exception as exp:
            print(f"Something went wrong with {exp}!!!!")
         
         
        
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

model = Model.from_pretrained("10epoch.trc", "config_file10epoch.cfg")
model = Model.from_config_file("write_it.cfg")

torch.manual_seed(0)
model([torch.randn(1, 1, 512), torch.tensor([0])])

model.save_model("10epoch")
        
"""


