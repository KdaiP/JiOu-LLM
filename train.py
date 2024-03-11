import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

from dataclasses import dataclass
from typing import Tuple
from random import randint

from tqdm import tqdm # ! pip install tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from dataset import LlamaDataset, collate_fn
from model import Transformer, ModelArgs

from scheduler import get_cosine_schedule_with_warmup

@dataclass
class TrainArgs:
    batch_size: int = 512
    num_workers: int = 4
    total_epochs: int = 3
    learning_rate: float = 1e-4
    warmup_steps: int = 20
    save_every_n_epoch: int = 1
    log_dir: str = 'runs'
    log_interval: int = 128

        
def ddp_setup(rank, world_size):
    """
    config:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    init_process_group(backend="gloo" if os.name == "nt" else "nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: Transformer,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        gpu_id: int,
        save_every: int,
        log_dir: str = "runs",
        log_interval: int = 128
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.log_interval = log_interval
        self.model = DDP(model, device_ids=[gpu_id])
        
        if self.gpu_id == 0:
            self.writer = SummaryWriter(log_dir=log_dir)

    def _run_batch(self, data: list, batch_idx, epoch):
        data, data_length = data
        self.optimizer.zero_grad()
        loss, acc = self.model.module.compute_loss(data, data_length)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        # 在DDP中，如果有层未参与梯度计算会报错。
        # 设置DDP(model, device_ids=[gpu_id]，find_unused_parameters=True)，然后检查哪些层没有参与梯度计算：
        # ls = [name for name,para in self.model.module.named_parameters() if para.grad==None]
        # print(ls)
        
        # tensorboard日志记录
        if self.gpu_id == 0 and batch_idx % 32 == 0:
            current_step = epoch * len(self.train_data) + batch_idx
            self.writer.add_scalar("loss", loss.item(), current_step)
            self.writer.add_scalar("acc", acc, current_step)
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], current_step)

    def _run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        train_loader = tqdm(self.train_data) if self.gpu_id == 0 else self.train_data
            
        for idx, data in enumerate(train_loader):
            self._run_batch([d.to(self.gpu_id, non_blocking=True) for d in data], idx, epoch)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = f"checkpoint_{epoch}.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs(rank, world_size, train_config: TrainArgs,  model_config: ModelArgs):
    train_dataset = LlamaDataset()  # load your dataset
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, num_workers=train_config.num_workers, pin_memory=True, batch_size=train_config.batch_size, sampler=DistributedSampler(train_dataset), persistent_workers=True, prefetch_factor=8)
    model = Transformer(model_config)  # load your model
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_config.warmup_steps, num_training_steps=train_config.total_epochs * len(train_loader))
    return train_dataset, train_loader, model, optimizer, scheduler



def main(rank: int, world_size: int, configs: Tuple[TrainArgs, ModelArgs]):
    train_config, model_config = configs
    ddp_setup(rank, world_size)
    train_dataset, train_loader, model, optimizer, scheduler = load_train_objs(rank, world_size, train_config, model_config)
    trainer = Trainer(model, train_loader, optimizer, scheduler, rank, train_config.save_every_n_epoch, train_config.log_dir, train_config.log_interval)
    trainer.train(train_config.total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    train_config = TrainArgs()
    model_config = ModelArgs()
    configs = (train_config, model_config)
    world_size = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    mp.spawn(main, args=(world_size, configs), nprocs=world_size)