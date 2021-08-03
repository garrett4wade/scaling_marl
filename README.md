## Usage

To initialize a learner node, run `bash run_learner_node.sh $(learner_node_idx)` and write corresponding learner configuration.
Learner configuration is a Python dict specifying the usage of each GPU on each learner node.
For example, if you have 2 learner nodes and you want to use GPU 0 and GPU 1 of node 0 for training policy 0 and
use GPU 2 of node 0 and GPU 0 of node 1 for training policy 1,
```yaml
learner_config:
  "0":
    "0": 0
    "1": 0
    "2": 1
  "1":
    "0": 1
```

To initialize a worker node, run `bash run_worker_node.sh $(worker_node_idx)`.
Number of worker nodes is specified by seg_addrs. For example, if you have 2 learner nodes and 2 worker nodes, seg_addrs should be like
```yaml
seg_addrs:
  - - learner0_addr:port0
    - learner0_addr:port1
  - - learner1_addr:port0
    - learner2_addr:port1
```
such that a worker can communicate with all learners. `len(seg_addrs)` equals to the number of learner nodes and
`len(seg_addrs[0])` equals to the number of worker nodes.

To run a single worker node to benchmark the throughput, comment [here](https://github.com/garrett4wade/scaling_marl/blob/master/system/policy_worker.py#L144)
and just run a worker node as normal.

# quick launch and monitor
Specify node numbers (for ssh), container names and learner/worker names in config, and run `python run.py` in this directory (run this outside of container with python2). Notice that `update.sh` update your code before running, you can write your own `update.sh` and uncomment code in `run.py` to auto update before run.

The monitor processes run on every container. The head monitor try to gather system information (RX, TX, cpu utilization) from other monitors every `interval` seconds, and print them out to log. If you want a graph for these information after a run, uncomment `m.summary()` line in `run.py`.