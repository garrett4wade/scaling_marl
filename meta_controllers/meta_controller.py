class MetaController:
    def __init__(self, cfg):
        self.cfg = cfg

        self.num_learner_tasks = cfg.num_policies
        self.num_worker_tasks = len(cfg.seg_addrs[0]) * cfg.num_tasks_per_node
    
    def reset(self):
        raise NotImplementedError

    def step(self, report):
        raise NotImplementedError
