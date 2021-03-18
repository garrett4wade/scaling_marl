import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x, choose=True):
        return self.fc1(x) if choose else self.fc2(x)


class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.choose = Model1()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x, choose=True):
        x = self.choose(x, choose)
        return self.fc1(x) if choose else self.fc2(x)


def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23458'
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    model = DDP(Model2().to(rank), device_ids=[rank], output_device=rank)
    a = torch.randn(128, 32).cuda()
    model(a)
    b = torch.randn(128, 64).cuda()
    model(b, False)
    dist.destroy_process_group()
    print('finish')


if __name__ == "__main__":
    world_size = 2
    procs = []
    for i in range(world_size):
        p = mp.Process(target=run, args=(i, world_size))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
