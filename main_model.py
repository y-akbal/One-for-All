import torch
from torch import nn as nn
from torch.nn import functional as F
from layers import attention_block, Upsampling, Linear, layernorm, LUpsampling, PUpsampling
import pickle
import typing
from tqdm import tqdm


class Model(nn.Module):
    def __init__(
        self,
        lags: int = 512,
        embedding_dim: int = 512,
        n_blocks: int = 25,
        pool_size: int = 4,
        number_of_heads=4,
        number_ts=25, ###This is needed for embeddings, you can have more than you need for prompt fine tuning 
        number_of_clusters=None,  ### number of clusters of times series
        conv_activation = None,
        conv_FFN_activation = nn.GELU(),
        conv_dropout_FFN = 0.2,
        conv_dropout_linear = 0.2,
        conv_FFN_bias = True,
        conv_FFN_expansion_size = 4,
        conv_bias = True,
        attention_FFN_dropout = 0.2,
        attention_head_dropout = 0.2,
        attention_projection_dropout = 0.2,
        attention_FFN_activation = nn.GELU(),
        attention_FFN_bias = True,
        attention_FFN_expansion_size = 4,
    ):
        assert (
            lags / pool_size
        ).is_integer(), "Lag size should be divisible by pool_size"
        super().__init__()
        self.lags = lags
        self.width = lags // pool_size
        self.embedding_dim = embedding_dim
        ###
        self.cluster_used = True if number_of_clusters is not None else False
        ###

        self.up_sampling = PUpsampling(
            lags=lags,
            d_out=self.embedding_dim,
            pool_size=pool_size,
            conv_bias= conv_bias, 
            dense_bias= conv_FFN_bias,
            conv_activation= conv_activation,
            FFN_activation=  conv_FFN_activation,
            num_of_ts=number_ts,
            num_of_clusters=number_of_clusters,
            FFN_expansion_size= conv_FFN_expansion_size,           
            dropout_FFN = conv_dropout_FFN,
            dropout_linear = conv_dropout_linear,
        )
        self.blocks = nn.Sequential(
            *(
                attention_block(
                    d_in = embedding_dim,
                    width=lags,
                    n_heads=number_of_heads,
                    dropout_FFN=attention_FFN_dropout,
                    activation=attention_FFN_activation,
                    expansion_size= attention_FFN_expansion_size,
                    bias_FFN=attention_FFN_bias,
                    att_head_dropout= attention_head_dropout,
                    projection_dropout=attention_projection_dropout
                )
                for _ in range(n_blocks)
            )
        )
        ### This dude is the final linear
        ### The same along all dimensions, we can replace it by an MLP
        self.Linear = nn.Sequential(*[layernorm(self.embedding_dim),
            Linear(self.embedding_dim, 1, bias = True, dropout=0.0)
        ])
        ###
        ### here is the config dict to be used
        self.config = {
            "lags": lags,
            "embedding_dim": embedding_dim,
            "n_blocks": n_blocks,
            "pool_size": pool_size,
            "number_of_heads": number_of_heads,
            "number_ts": number_ts,
            "number_of_clusters": number_of_clusters,
            "conv_activation": conv_activation,
            "conv_FFN_activation": conv_FFN_activation,
            "conv_dropout_FFN": conv_dropout_FFN,
            "conv_dropout_linear": conv_dropout_linear,
            "conv_FFN_bias": conv_FFN_bias,
            "conv_FFN_expansion_size": conv_FFN_expansion_size,            
            "conv_bias": conv_bias,
            "attention_head_dropout":attention_head_dropout,
            "attention_projection_dropout": attention_projection_dropout,
            "attention_FFN_dropout": attention_FFN_dropout,
            "attention_FFN_activation": attention_FFN_activation,
            "attention_FFN_bias": attention_FFN_bias,
            "attention_FFN_expansion_size": attention_FFN_expansion_size,
        }

    def forward(self, x:tuple[torch.Tensor, torch.Tensor]):
        ## Here we go with upsampling layer
        if self.cluster_used:
            x, tse_embedding, cluster_embedding = x[0].unsqueeze(-2), x[1].unsqueeze(-1), x[2]
            x = self.up_sampling((x, tse_embedding, cluster_embedding))
            
        else:
            x_, tse_embedding = x[0].unsqueeze(-2), x[1].unsqueeze(-1)
            
            x = self.up_sampling((x_, tse_embedding))
            
        ## Concatted transformer blocks
        ###

        return self.Linear(self.blocks(x)).squeeze(-2)

    @classmethod
    def from_config_file(cls, config_file):
        with open(config_file, mode="rb") as file:
            dict_ = pickle.load(file)
        if isinstance(dict_, dict):
            return cls(**dict_)
        else:
            raise ValueError("The pickled file should contain a config dictionary")

    ### These dudes stay here for future versions ### 
    ###  mostly for inference using single gpu!!! ###
    @classmethod
    def from_pretrained(cls, file_name):
        try:
            dict_ = torch.load(file_name)
            config = dict_["model_config"]
            state_dict = dict_["model_state_dict"]
            model = cls(**config)

            model.load_state_dict(state_dict)
            print(
                f"Model loaded successfully!!!! The current configuration is {config}"
            )

        except Exception as e:
            print(f"Something went wrong with {e}")
        return model

    @classmethod
    def from_data_class(cls, data_class):
        if isinstance(data_class, dict):
            return cls(**data_class)
        else:
            return cls(**data_class.__dict__)
    
    def __init__weights__(self, seed = 0):
        ## To be used for better initialization depending on the depth of the layers!!!
        ## Lower layers may require a bit larget gradients!!!
        pass


    def save_model(self, file_name = None):
        fn = "Model" if file_name == None else file_name
        model = {}
        model["model_state_dict"] = self.state_dict()
        model["model_config"] = self.config
        try:
            torch.save(model, f"{fn}")
            print(
                f"Model saved succesfully, see the file {fn} for the weights and config file!!!"
            )
        except Exception as exp:
            print(f"Something went wrong with {exp}!!!!!")

    @torch.no_grad()
    def generate(self, 
                 x_init:tuple[torch.Tensor,torch.Tensor], 
                 horizon:int = 10,              
                 ):
        ## Batched long term forcast --- 

        device = f"cuda:{x_init[0].get_device()}" if x_init[0].get_device() >= 0 else "cpu" ## Get the device

        B, L = x_init[0].shape
        if L + horizon > self.lags:
            print(f"The model can handle long term forcasts up to horizon {self.lags}, while yours {L+horizon}. The begining of the series will be clipped!!!")

        horizon_predictions = torch.empty(B, L+horizon, device = device) ## In malloc we trust!!!
        horizon_predictions[:, :L] = x_init[0]

        tqdm_range =  tqdm(range(horizon))
        
        for i in tqdm_range:
            if L+i <= self.lags:
                next_lag = self([horizon_predictions[:, :L+i], x_init[1]])[:, -1]
            else:
                next_lag = self([horizon_predictions[:, -self.lags +(L+i):L+i], x_init[1]])[:, -1]

            horizon_predictions[:, L+i] = next_lag

        return horizon_predictions



"""
torch.manual_seed(0)
model = Model(lags = 128, embedding_dim= 128).cuda()
model([torch.randn(1, 4).cuda(), torch.tensor([1]).cuda()])
torch.manual_seed(0)
q = torch.randn(100, 50).cuda()
model.generate([q, torch.tensor([3 for i in range(100)]).cuda()], horizon = 150).shape




"""

"""
x = Model()
q = torch.torch.distributions.Uniform(low=-1, high=1).sample((1, 512))
x.eval()
x([q, torch.tensor([5])]).std()
"""
