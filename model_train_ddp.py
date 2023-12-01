import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        gpu_id: int,
        save_every: int,
        val_loss_logger = None,
        train_loss_logger = None,
        val_accuracy_logger = None,
        compile = False,
        use_wnb = False,
        use_DDP = True,
    ) -> None:
        self.gpu_id = gpu_id
        self.model_config = model.config
        self.model = model.to(gpu_id)
        if use_DDP:
            self.model = DDP(self.model, device_ids=[gpu_id])
        if compile:
            self.model = torch.compile(self.model)
        ##
        self.train_data = train_data
        self.val_data = val_data
        ##
        self.optimizer = optimizer
        self.scheduler = scheduler
        ##
        self.save_every = save_every
        ##
        self.val_loss_logger = val_loss_logger
        self.train_loss_logger = train_loss_logger
        self.val_accuracy_logger = val_accuracy_logger
        self.use_wnb = use_wnb
        ##
        ##
        self.autocast = torch.autocast
        self.scaler = torch.cuda.amp.GradScaler()
        ## training details ##
        self.epoch = 1

        try:
            self._load_checkpoint("checkpoint.pt")
            print(f"Started from {self.epoch} -- where stopped last time!!!")
        except Exception as e:
            print(f"There is a problem with loading the model weights and the problem is: {e}")
        
    def _run_batch(self, source, targets, i):
        ### All the things like low precision training will happen here!!!
        self.model.train() ## Model in train mode!!!
        self.optimizer.zero_grad()
        with self.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model(source, task = None)
            loss = F.mse_loss(output, targets)

        if i % 100 == 0:
            print(f"loss {loss.item()}, {i} batch, from gpu {self.gpu_id} ")
        ## Update the the gradients here!!! ## 
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        ### We log the loss ### 

 
    def _run_epoch(self, epoch, report_in_every = 100):
        # b_sz = len(next(iter(self.train_data))[0])
        if epoch % report_in_every == 0:
            print(f"[GPU{self.gpu_id}] Epoch {epoch}")
        self.train_data.sampler.set_epoch(epoch)

        for i, (source, targets) in enumerate(self.train_data):
            source = source.to(self.gpu_id, non_blocking=True)
            targets = targets.to(self.gpu_id, non_blocking=True)
            self._run_batch(source, targets, i)

    def train(self, max_epochs: int):
        for epoch in range(self.epoch, max_epochs):
            self._run_epoch(epoch)
            self.epoch = epoch #update epoch!!!             
            if self.gpu_id == 0 and epoch % self.save_every == 0:
               self._save_checkpoint(epoch)
            self.validate()

    def validate(self):
        if self.gpu_id == 0:
            print("Validation started!!!")
        self.model.eval()
        with torch.no_grad():  ## block tracking gradients
            for source, targets, _ in self.val_data:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                output = self.model(source)  
                loss = F.mse_loss(output, targets)
                self.val_loss_logger.update(loss.item())
                
                #print(self.val_accuracy_logger.update(accuracy.item()))
                

            self.val_loss_logger.all_reduce()
            
            if self.gpu_id == 0:
                print(self.val_loss_logger.get_avg_loss(), self.val_accuracy_logger.accuracy)
                
                self.val_loss_logger.reset()
                self.val_accuracy_logger.reset()    

    ## Some tools ## 
    def _load_checkpoint(self, checkpoint_file):
        model_dict = torch.load(checkpoint_file)
        ### Now the state dict are obtained below ###
        model_state_dict = model_dict["model_state_dict"]
        model_optimizer_state = model_dict["optimizer_state"]
        ### ---Let's load the model states--- ###
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(model_optimizer_state)
        self.epoch = model_dict["epoch"]
        print("Loaded the new model!!!!")
 
    def _save_checkpoint(self):
        ### This are the necessary steps to recover the model from the pickled file!!!
        model_weights = self.model.state_dict()
        model_config = self.model_config
        optimizer_state = self.optimizer.state_dict()

        checkpoint = {
                      "model_state_dict":model_weights,
                      "model_config":model_config,
                      "optimizer_state":optimizer_state,
                      "epoch":self.epoch
                    }
        try:
            PATH = "checkpoint.pt"
            torch.save(checkpoint, PATH)        
            print(f"Epoch {self.epoch} | Training checkpoint saved at {PATH}")
        except Exception as exp:
            print(f"Something went wrong with {exp}!!!")
    


if __name__ == "__main__":
    print("Ok boomer!!!")