import psutil
import multiprocessing as mp
import time
from monitor.network import Pusher, Puller
import subprocess
from copy import copy
import os
from collections import deque
from copy import copy, deepcopy

import matplotlib.pyplot as plt
import numpy as np
import random

class Monitor():
    # start/stop monitor process
    # communicate with central control, keep heartbeat 
    # central control restart this if main process does not exists. (or restart if fault)
    # detect anomaly
    # start if process dont exists, restart if anomaly detected
    def __init__(self, config):
        self.monitor_process = None
        config = self._init_config(config)
        # for debug purpose
        self.total_rounds = config.total_rounds
        # timeout
        # port
        # times
        # self.uptime = 0
        self.interval = config.interval
        self.verbose = config.verbose

        # names and addresses
        self.name = config.name
        self.nodes = config.nodes

        self.is_head = config.is_head
        self._info_queue = mp.Queue()
        self.info_list = []

        # only for head node. string list of head self update because of python reference
        self.info_store = {}

        for node in self.nodes:
            self.info_store[node] = []

        # self.type = config.type
        #self._worker_process = None
        self.monitor_process = None

        self.monitor_pid = [os.getpid()]

        # calculation var for print info
        self.print_info = {}
        self.last_info_string = {}

        # child nodes
        if not self.is_head:
            self.pusher = Pusher(config.puller_address, config.puller_port)

        # head node
        if self.is_head:
            self.puller = Puller(config.puller_port)
        # monitor network
    
    def _init_config(self, config):
        config.nodes = config.nodes.split(",")
        return config

    def get_info_string(self, info):
        if not info["name"] in self.print_info:
            self.print_info[info["name"]] = [0, 0, 0, 0, 0] # self.last_bs, self.last_br, self.last_ps, self.last_pr, self.last_time = 0, 0, 0, 0, 0
        
        pinfo = self.print_info[info["name"]]
        info_string = "Node name: {} (Timestamp: {})\n".format(info["name"], info["timestamp"])
        info_string += "Total CPU percent: {:.2%}, used cpu percent: {:.2%}\n" \
                        .format(info["total_cpu_percent"], info["total_cpu_percent"]/(psutil.cpu_count()))
        info_string += "Total RSS: {}\n".format(info["total_rss"])
        info_string += "Processes: \n"
        if self.verbose:
            for t in info:
                try:
                    # print("checkint:", t)
                    int(t)
                except ValueError:
                    # print("not int:", t)
                    continue
                pid = t
                if pid in info["monitor_pid"]:
                    info_string += "MONITOR PROCESS:\n"
                infod = info[t]
                if info["name"] == self.name:
                    info_string += "pid: {}, process name: {}, cpu num: {}, cpu percent: {:2%}, status: {}, mem rss: {}, mem vms: {}, mem shared: {}\n"\
                            .format(pid, infod["name"], infod["cpu_num"], infod["cpu_percent"], infod["status"], infod["memory_full_info"].rss, \
                                infod["memory_full_info"].vms, infod["memory_full_info"].shared)
                else:
                    info_string += "pid: {}, process name: {}, cpu num: {}, cpu percent: {:2%}, status: {}, mem rss: {}, mem vms: {}, mem shared: {}\n"\
                            .format(pid, infod["name"], infod["cpu_num"], infod["cpu_percent"], infod["status"], infod["memory_full_info"][0], \
                                infod["memory_full_info"][1], infod["memory_full_info"][2])
                
        info_string += "network bytes sent: {} , bytes recv: {}, packets sent: {}, packets recv:{}, errin: {}, errout: {}, dropin: {}, dropout:{}\n"\
                .format(info["bytes_sent"], info["bytes_recv"], info["packets_sent"], info["packets_recv"], \
                    info["errin"], info["errout"], info["dropin"], info["dropout"])

        if pinfo[4] != 0:
            try:
                avg_tx = round(((info["bytes_sent"] - pinfo[0]) * 8/(info["timestamp"] - pinfo[4]))/ 10**9, 2)
                avg_rx = round(((info["bytes_recv"] - pinfo[1]) * 8/(info["timestamp"] - pinfo[4]))/ 10**9, 2)
                avg_ps = round((info["packets_sent"] - pinfo[2])/(info["timestamp"] - pinfo[4]), 2)
                avg_pr = round((info["packets_recv"] - pinfo[3])/(info["timestamp"] - pinfo[4]), 2)
                info_string += "(since last info) TX: {} Gbits/s, RX: {} Gbits/s, packets sent rate: {} pkts/s, packets recv rate:{} pkts/s\n"\
                        .format(avg_tx, avg_rx, avg_ps, avg_pr)
                        
                info["RX"] = avg_rx
                info["TX"] = avg_tx
            except ZeroDivisionError:
                print("Info for node \"%s\" not updated in this round, showing latest info." % info["name"])
                return self.last_info_string[info["name"]]

        self.last_info_string[info["name"]] = info_string
        self.print_info[info["name"]] = [info["bytes_sent"], info["bytes_recv"], info["packets_sent"], info["packets_recv"], info["timestamp"]]
        
        return info_string

    def update_info(self):
        # prob info and print
        info = None
        while not self._info_queue.empty():
            info, curtime = self._info_queue.get()
            if info:
                info["timestamp"] = curtime
                info["name"] = self.name
                info["monitor_pid"] = self.monitor_pid
                self.info_list.append(info)
            # debug
            # print(self.get_info_string(info))
    
    # only called on headnode, print latest info on all nodes
    def print_latest_info(self, round_):
        print("\nRound %d info print:"  % round_)
        for n, store in self.info_store.items():
            if len(store) > 0:
                print(self.get_info_string(store[-1]) + "\n")
            print("Tracking storage length: %s: %d" % (n, len(store)))
                    

    # gather info strings, only can be called on head node
    def collect_round(self):
        stop = False
        #self.nodes
        no_receive = copy(self.nodes)
        no_receive.remove(self.name)
        round_received = {}
        count = 0
        while not stop:
            info = self.puller.pull()
            if info:
                name = info["name"]
                if not name in round_received:
                    round_received[name] = info
                    no_receive.remove(name)
                else:
                    if round_received[name]["timestamp"] <= info["timestamp"]:
                        round_received[name] = info
                    
            if len(no_receive) == 0:
                stop = True
                return round_received
            if count > (len(self.nodes) - 1) * 5:
                stop = True
                return round_received, no_receive
            # print(count, no_receive)
            count += 1

    def push_round(self):
        pushed = False
        while not pushed:
            try:
                self.pusher.push(self.info_list[-1])
                pushed = True
            except IndexError:
                continue              

    def stuff(self, y1, y2):
        assert len(y1) > len(y2)
        mul = len(y1) // len(y2)
        mod = len(y1) % len(y2)
        y = []
        modsample = random.sample(range(len(y2)), mod)
        for i, vy in enumerate(y2):
            for _ in range(mul):
                y.append(vy)
            if i in modsample:
                y.append(vy)
        return y
    
    def summary(self, nameset = None):
        if not self.is_head:
            return

        if nameset == None:
            nameset = self.nodes

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

        ys1 = [[] for _ in nameset]
        for i, n in enumerate(nameset):
            infos = self.info_store[n]
            for info in infos:
                ys1[i].append(info["total_cpu_percent"])
        
        if len(nameset) > 1:
            for i in range(1, len(ys1)):
                ys1[i] = self.stuff(ys1[0], ys1[i])
        
        for i, n in enumerate(nameset):
            ax1.plot(range(len(ys1[0])), ys1[i], label = n)

        ax1.title.set_text("CPU utilization")
        # network TX
        ys2 = [[] for _ in nameset]
        for i, n in enumerate(nameset):
            infos = self.info_store[n]
            for info in infos:
                try:
                    ys2[i].append(info["TX"])
                except:
                    continue

        if len(nameset) > 1:
            for i in range(1, len(ys2)):
                ys2[i] = self.stuff(ys2[0], ys2[i])
        
        for i, n in enumerate(nameset):
            ax2.plot(range(len(ys2[0])), ys2[i], label = n)

        ax2.title.set_text("TX")

        # network RX
        ys3 = [[] for _ in nameset]
        for i, n in enumerate(nameset):
            infos = self.info_store[n]
            for info in infos:
                try:
                    ys3[i].append(info["RX"])
                except:
                    continue
        
        if len(nameset) > 1:
            for i in range(1, len(ys3)):
                ys3[i] = self.stuff(ys3[0], ys3[i])
        
        for i, n in enumerate(nameset):
            ax3.plot(range(len(ys3[0])), ys3[i], label = n)

        ax3.title.set_text("RX")

        plt.savefig("/workspace/figs/summary.png")
    
    def run(self):
        round_ = 0
        while round_ < self.total_rounds:
            self.update_info()
            if len(self.info_list) > 0:
                if self.is_head:
                    # for head node, gather lattest info for all other nodes
                    ret = self.collect_round()

                    if type(ret) == tuple:
                        round_received, dead = ret
                    else:
                        round_received = ret
                    
                    # print(round_received)
                    ## deal with cases
                    for k in round_received:
                        last_timestamp = 0
                        if len(self.info_store[k]) > 0:
                            last_timestamp = self.info_store[k][-1]["timestamp"]
                        if round_received[k]["timestamp"] > last_timestamp:
                            self.info_store[k].append(deepcopy(round_received[k]))
                    if len(self.info_list) > 0:
                        self.info_store[self.name].append(deepcopy(self.info_list[-1]))

                else:
                    self.push_round()

            # for debug purpose
            self.print_latest_info(round_)

            time.sleep(self.interval)
            round_ += 1
        print("FINISHED ALL ROUNDS, BEGIN TO SUMMARY AND QUIT.")

    def start_monitor(self):
        # start local moniter
        # do nothing if monitor already running
        if not self.monitor_process == None and self.monitor_process.is_alive():
            return

        self.monitor_process = MonitorProcess(self._info_queue)
         
        self.monitor_process.start()
        self.monitor_pid.append(self.monitor_process.pid)
        print("MONITOR PID {}".format(str(self.monitor_pid)))


    def terminate_monitor(self):
        if not self.monitor_process == None:
            self.monitor_process.terminate()
        else:
            print("Monitor not started. Terminate failed.")

    def close(self):
        time.sleep(1)
        if not self.monitor_process == None:
            self.monitor_process.close()

class MonitorProcess(mp.Process):
    def __init__(self, info_queue, interval:float = 1.0):
        super().__init__()
        self._start_time = time.time() 
        
        # self.procs = None
        # record critical cpu utils
        self.info_queue = info_queue

        self._interval = interval

    def run(self):
        while True:
            curtime = int(time.time())
            try:
                self.info_queue.put((self._processes_update(), curtime))
            except:
                print("Failed to acquire info, retry.")
            # print(self.info)
            time.sleep(self._interval)

    def _processes_update(self):
        # update new processes (psutil.Process instances)
        processes = psutil.process_iter()
        infod = {}
        # update critical data
        for p in processes:
            infod[p.pid] =  { "name": p.name(),
                              "cpu_num": p.cpu_num(),
                              "cpu_percent": p.cpu_percent()/psutil.cpu_count(),
                              "status": p.status(),
                              "memory_full_info": p.memory_full_info()
                            }
        
        infod["total_cpu_percent"] = 0
        infod["total_rss"] = 0 
        for k in infod: 
            if type(k) == str:
                continue
            infod["total_cpu_percent"] += infod[k]["cpu_percent"]
            infod["total_rss"] += infod[k]["memory_full_info"].rss
            # parse info into total info of python process
        # print(infod)
        # measure network
        snetio_ = psutil.net_io_counters()
        infod["bytes_sent"] = snetio_.bytes_sent
        infod["bytes_recv"] = snetio_.bytes_recv
        infod["packets_sent"] = snetio_.packets_sent
        infod["packets_recv"] = snetio_.packets_recv
        infod["errin"] = snetio_.errin
        infod["errout"] = snetio_.errout
        infod["dropin"] = snetio_.dropin
        infod["dropout"] = snetio_.dropout
        
        return infod

    @property
    def interval(self):
        print("Monitor refresh interval: %d seconds." % self._interval)
        return self._interval

    @interval.setter
    def interval(self, v:float):
        if v < 0:
            v = 10
        print("Set refresh interval to %d seconds." % self._interval)
        self._interval = v
        