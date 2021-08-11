from utils.utils import TaskType
from meta_controllers.meta_controller import MetaController
import random


class NaiveMetaController(MetaController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recent_elapsed_training_step = 0

    def reset(self, learner_socket_ident, worker_socket_ident):
        super().reset(learner_socket_ident, worker_socket_ident)
        tasks = []
        for i in range(self.num_learner_tasks):
            tasks.append([self.learner_socket_ident[i], str(TaskType.TRAIN).encode('ascii')])

        for i in range(self.num_worker_tasks):
            tasks.append([self.worker_socket_ident[i], str(TaskType.ROLLOUT).encode('ascii')])

        return tasks

    def step(self, report):
        self.recent_elapsed_training_step += 1
        if self.cfg.use_eval and self.recent_elapsed_training_step >=self.cfg.eval_interval:
            self.recent_elapsed_training_step = 0
            dst_ident = random.choice(self.worker_socket_ident)
            return [[dst_ident, str(TaskType.EVALUATION).encode('ascii')], [dst_ident, str(TaskType.ROLLOUT).encode('ascii')]]
        else:
            return []
