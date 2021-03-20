import torch.multiprocessing as mp
# import multiprocessing as mp
import time
import torch.nn as nn
import torch

if __name__ == "__main__":
    q = mp.Queue(8)

    def run(q):
        while True:
            ll = nn.Linear(1024, 1024).to(1)
            q.put(ll.state_dict())

    p = mp.Process(target=run, args=(q, ))
    p.start()

    ts = []
    l2 = nn.Linear(1024, 1024).to(2)
    while True:
        if q.qsize() == 0:
            time.sleep(1)
        tik = time.time()
        l2.load_state_dict(q.get())
        tok = time.time()
        ts.append(tok - tik)
        if len(ts) >= 20:
            print(sum(ts))
            ts = []
