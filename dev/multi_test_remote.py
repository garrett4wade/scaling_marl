from torch.distributed import rpc
import torch.multiprocessing as mp
import argparse
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--world_size', type=int)
parser.add_argument('--offset', type=int)

rpc_init_method = 'tcp://192.168.1.103:6391'
rpc_opt = rpc.TensorPipeRpcBackendOptions(init_method=rpc_init_method, rpc_timeout=300)


def task():
    a = np.random.randn(3, 4)
    return a**2


def run(rank, world_size):
    rpc.init_rpc('agent_' + str(rank), rank=rank, world_size=world_size, rpc_backend_options=rpc_opt)
    print('successfully setup rank {}'.format(rank))

    time.sleep(2)

    rpc.shutdown()
    print('finish {}!'.format(rank))


if __name__ == "__main__":
    args = parser.parse_args()
    procs = []
    for i in range(args.offset, args.world_size):
        p = mp.Process(target=run, args=(i, args.world_size))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
