# simple network utils based on zmq

import zmq
from threading import Thread
from copy import copy
from collections import deque
import time

# one client to multiple servers, REQ/REP model

class Server:
    def __init__(self, port, maxbuf = 1024):
        self.context = zmq.Context()    
        self.socket = self.context.socket(zmq.REP)
        
        self.socket.bind("tcp://*:%d" % port)

        self.maxbuf = maxbuf
        self.msgbuf = []

        self.thread = Thread(target = self._run)
        self.thread.start()

    def _run(self):
        while True:
            message = socket.recv()
            if len(self.msgbuf) < self.maxbuf:
                self.msgbuf.append(message)
                self.socket.send("ack")
            else:
                self.socket.send("full")
    
    def check(self):
        return self.msgbuf
    
    def get(self):
        tmp = copy(self.msgbuf)
        self.msgbuf.clear()
        return tmp
            
class Client:
    def __init__(self, addresses, timeout):
        self.context = zmq.Context()

        self.sockets = {addr: self.context.socket(zmq.REQ) for addr in addresses}
        for addr in addresses:
            self.sockets[addr].connect(addr)
            self.sockets[addr].setsockopt(zmq.RCVTIMEO, timeout)
    
    def send(self, msg, addr):
        self.sockets[port].send(msg)
        reply = self.socket[port].recv()
        if reply == "full":
            raise FullBufferError("Server side message buffer is full.")

            
class FullBufferError(Exception):
    pass


# recv timeout ?


# push / pull
class Pusher:
    def __init__(self, address, port, is_json = True):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect("tcp://%s:%d" % (address, port))

        self.json = is_json

    def push(self, msg):
        if self.json:
            self.socket.send_json(msg)
            # print("--------------\nNetwork debug, json msg sent: %s\n-------------" % str(msg))
        else:
            self.socket.send(msg)


class Puller:
    def __init__(self, port, is_json = True):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind("tcp://*:%d" % port)
        self.dq = deque(maxlen = 1024)
        
        self.json = is_json

        self.thread = Thread(target = self.run)
        self.thread.start()

    def run(self):
        while True:
            if self.json:
                msg = self.socket.recv_json()
                self.dq.append(msg)
                # print("--------------\nNetwork debug, json msg recv: %s\n-------------" % str(msg))
            else:
                self.dq.append(self.socket.recv())

    def pull(self):
        start_ = time.time()
        while True:
            if time.time() - start_ > 1:
                return None
            try:
                first = self.dq.popleft()
                break
            except IndexError:
                continue
        
        return first
