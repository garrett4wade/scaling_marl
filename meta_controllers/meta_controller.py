class MetaController:
    def __init__(self, cfg):
        self.cfg = cfg

        self.num_learner_tasks = cfg.num_policies
        self.num_worker_tasks = len(cfg.seg_addrs[0]) * cfg.num_tasks_per_node

        self.learner_socket_ident = [None for _ in range(self.num_learner_tasks)]
        self.worker_socket_ident = [None for _ in range(self.num_worker_tasks)]

    def reset(self, learner_socket_ident, worker_socket_ident):
        self.learner_socket_ident = learner_socket_ident
        self.worker_socket_ident = worker_socket_ident

    def step(self, report):
        raise NotImplementedError
