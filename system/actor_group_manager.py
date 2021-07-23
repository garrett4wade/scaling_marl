import multiprocessing as mp


class ActorGroupManager(mp.Process):
    def __init__(self, cfg, idx, policy_queue, group_semaphores):
        super().__init__()
        self.cfg = cfg

        self.group_idx = idx
        self.num_actor_groups = self.cfg.num_actors // self.cfg.actor_group_size
        self.policy_queue = policy_queue
        self.group_semaphores = group_semaphores
        self.num_splits = len(self.group_semaphores[0])

        self.daemon = True

    def run(self):
        split_idx = 0
        while True:
            for s in self.group_semaphores:
                s[split_idx].acquire()
            self.policy_queue.put(split_idx * self.num_actor_groups + self.group_idx)

            split_idx = (split_idx + 1) % self.num_splits
