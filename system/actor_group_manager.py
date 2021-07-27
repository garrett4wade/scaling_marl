import multiprocessing as mp


class ActorGroupManager(mp.Process):
    def __init__(self, cfg, idx, policy_queues, group_semaphores):
        super().__init__()
        self.cfg = cfg

        self.group_idx = idx
        self.num_actor_groups = (self.cfg.num_actors // self.cfg.actor_group_size // self.cfg.num_tasks_per_node)
        self.policy_queues = policy_queues
        self.group_semaphores = group_semaphores
        self.num_splits = len(self.group_semaphores[0])

        self.daemon = True

    def run(self):
        split_idx = 0
        while True:
            for s in self.group_semaphores:
                s[split_idx].acquire()
            for policy_queue in self.policy_queues:
                policy_queue.put(split_idx * self.num_actor_groups + self.group_idx)

            split_idx = (split_idx + 1) % self.num_splits
