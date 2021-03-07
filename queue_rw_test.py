# from queue import Queue
from multiprocessing import Queue
import numpy as np
import time

if __name__ == "__main__":
    ts = []
    q = Queue(128)
    for _ in range(10):
        a = np.random.randn(1024, 1024, 10)
        tik = time.time()
        q.put(a)
        ts.append(time.time() - tik)
    print("put time: {}".format(np.mean(ts)))

    ts = []
    for _ in range(10):
        tik = time.time()
        a = q.get()
        ts.append(time.time() - tik)
    print("get time: {}".format(np.mean(ts)))
