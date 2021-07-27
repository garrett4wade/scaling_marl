from utils.utils import TaskType
from meta_controllers.meta_controller import MetaController


class NaiveMetaController(MetaController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.cfg.num_policies == 1

    def reset(self):
        task = []
        for _ in range(self.num_learner_tasks):
            task.append(str(TaskType.TRAIN).encode('ascii'))

        for _ in range(self.num_worker_tasks):
            task.append(str(TaskType.ROLLOUT).encode('ascii'))

        return task

    def step(self, report):
        return [None for _ in range(self.num_learner_tasks + self.num_worker_tasks)]
