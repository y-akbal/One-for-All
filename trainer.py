import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import os
from training_tools import loss_track

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        save_every: int,
        max_epochs: int,
        snapshot_dir:str = "model",
        snapshot_name:str = "model.pt",
        compile_model:bool = False,
        use_wnb:bool = True,
        val_loss_logger = None,
        train_loss_logger = None,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model_config = model.config
        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        if compile_model:
            self.model = torch.compile(self.model)
        ##
        self.train_data = train_data
        self.val_data = val_data
        ##
        self.optimizer = optimizer
        self.scheduler = scheduler
        ##
        self.max_epochs = max_epochs
        self.save_every = save_every
        self.snapshot_dir = snapshot_dir
        self.snapshot_name = snapshot_name
        self.PATH = os.path.join(self.snapshot_dir, self.snapshot_name) if os.path.exists(self.snapshot_dir) else None
        ##
        self.val_loss_logger = val_loss_logger
        self.train_loss_logger = train_loss_logger
        ##
        self.use_wnb = use_wnb
        ##
        ##
        self.autocast = torch.autocast
        self.scaler = torch.cuda.amp.GradScaler()
        ## training details ##
        self.epoch = 1

        try:
            self._load_checkpoint(self.PATH)
            print(f"Training is continued from epoch {self.epoch}!!!")
        except Exception as e:
            print(f"There is a problem with loading the model weights and the problem is: {e}")
        
    def _run_batch(self, source, cls_):
        ### All the things like low precision training will happen here!!!
        N = source.shape[1]
        X, y = source[:, :-1], source[:,4:N:4]
        ## -- ##        
        self.optimizer.zero_grad()
        with self.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model([X.unsqueeze(-2), cls_.unsqueeze(-1)])
            loss = F.mse_loss(output.squeeze(), y)
        ## Log the loss

        ## Update the weights
        self.train_loss_logger.update(loss)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
    
    def _run_epoch(self, epoch):
        #if epoch % report_in_every == 0 and self.gpu_id == 0:
        #    print(f"[GPU{self.gpu_id}] Epoch {epoch}")
        self.train_data.sampler.set_epoch(epoch)
        for i, (source, cls_, file_name) in enumerate(self.train_data):
            source, cls_ = map(lambda x: x.to(self.gpu_id, non_blocking=True), [source, cls_])

            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            
            init_start.record() ## How much time we spent!!!
            self._run_batch(source, cls_)
            init_end.record() ## let's record it now!!!
            torch.cuda.synchronize() 
            if i % 250 == 0:
                print(f"{i}th batch just passed!!! loss is {self.train_loss_logger.loss} lr is {self.scheduler.get_last_lr()}, Time for single batch {init_start.elapsed_time(init_end) / 1000}")
                
    
    def train(self):
        for epoch in range(self.epoch, self.max_epochs):
            ## Do training on one epoch --
            if self.gpu_id == 0:
                print(f"Epoch {self.epoch}")
            self.model.train()
            self._run_epoch(epoch)
            self.scheduler.step()
            self.train_loss_logger.all_reduce()
            self.train_loss_logger.reset()
            ## Some saving --
            self.epoch += 1 #update epoch!!!             
            ## Let's do some validation
            if self.gpu_id == 0 and (epoch - 1) % self.save_every == 0:
               self._save_checkpoint()
            ## Let's start the validation
            self.validate()

    def validate(self):
        if self.gpu_id == 0:
            print("Validation started!!!")
        self.model.eval()
        with torch.no_grad():  ## block tracking gradients
            for source, targets, cls_, file_name in self.val_data:
                source, targets, cls_ = map(lambda x: x.to(self.gpu_id, non_blocking=True), [source, targets, cls_])
                output = self.model([source.unsqueeze(-2), cls_.unsqueeze(-1)])
                loss = F.mse_loss(output.squeeze(), targets)
                self.val_loss_logger.update(loss.item())
            self.val_loss_logger.all_reduce()            
            if self.gpu_id == 0:
                print(self.val_loss_logger.loss)
            self.val_loss_logger.reset()
                

    ## Some tools ## 
    def _load_checkpoint(self, checkpoint_file):
        state_dict = torch.load(checkpoint_file)
        ## Where we stopped at?
        keys = ["epoch", "model_state_dict", "optimizer_state","scheduler_state"]
        self.epoch, model_state_dict, optimizer_state, scheduler_state = map(lambda x: state_dict[x], keys)
        ### ---Let's load the model states--- ###
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state)
        self.scheduler.load_state_dict(scheduler_state)
        print(f"Loaded the new model succesfully!!! The training will continue from epoch {self.epoch}")
 
    def _save_checkpoint(self):
        ### This are the necessary steps to recover the model from the pickled file!!!
        ### prodived that you do training on a single GPU.
        model_weights = self.model.state_dict()
        model_config = self.model_config
        optimizer_state = self.optimizer.state_dict()
        scheduler_state = self.scheduler.state_dict()
        
        checkpoint = {
                      "model_state_dict":model_weights,
                      "model_config":model_config,
                      "optimizer_state":optimizer_state,
                      "scheduler_state": scheduler_state,
                      "epoch":self.epoch
                    }
        if self.PATH is None:
            os.mkdir(self.snapshot_dir)
            self.PATH = os.path.join(self.snapshot_dir,self.snapshot_name)            
        try:
            torch.save(checkpoint, self.PATH)        
            print(f"Epoch {self.epoch} | Training checkpoint saved at {self.PATH}")
        except Exception as exp:
            print(f"Something went wrong with {exp}!!!")
    


if __name__ == "__main__":
    print("Ok boomer!!!")