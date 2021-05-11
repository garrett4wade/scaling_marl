import zmq
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import time
from collections import deque


socket = zmq.Context().socket(zmq.ROUTER)
socket.bind('tcp://*:12345')

ts = []
frames = 0
frame_tik = time.time()
while True:
    time.sleep(.05)
    msg_tik = time.time()
    msg = socket.recv_multipart()
    print('receive some message!')

    socket.send_multipart([msg[0], msg[1], b'ok'])

    if len(msg) > 3:
        msg = msg[2:]
        tik = time.time()
        assert len(msg) % 2 == 0
        seg_dict = {}
        for i in range(len(msg) // 2):
            k, v = msg[2*i], msg[2*i+1]
            array = np.frombuffer(memoryview(v), dtype=np.float32)
            seg_dict[k.decode('ascii')] = array
        
        frames += 400
        ts.append(time.time() - tik)

    if len(ts) >= 10:
        fps = frames / (time.time() - frame_tik)
        print('recv msg', sum(ts) / len(ts), 'FPS {:.2f}'.format(fps))
        ts = []
        frames = 0
        frame_tik = time.time()
