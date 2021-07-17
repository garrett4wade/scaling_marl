## Usage

To initialize a learner node, run `bash run_learner_node.sh $(learner_node_idx) $(trainer_indices_on_this_node)`, where `$(trainer_indices_on_this_node)` is sperated by commas.
For example, if you have 2 learner nodes and you want to initialize 4 trainers in total, 2 for each node, then
```python
# on the first node
bash run_learner_node.sh 0 0,1

# on the second node
bash run_learner_node.sh 1 2,3
```

To initialize a worker node, run `bash run_worker_node.sh $(worker_node_idx)`.
To run a single worker node to benchmark the throughput, comment [here](https://github.com/garrett4wade/scaling_marl/blob/master/system/policy_worker.py#L127)
and just run a worker node as normal.

Note that the usage must be in correspondance with the configuration file.
The number of `model_weights_addrs` must equal to the number of learner nodes and the number of `seg_addrs` must
equal to the number of worker nodes. `num_trainers` must equal to the number of trainers in all learner nodes.
Currently there're few assertions about the usage. Need to fix it in the future.
