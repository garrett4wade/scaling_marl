import torch.multiprocessing as mp
import time

import torch
import zmq

import numpy as np
from system.transmitter import Transmitter
from system.worker_task import WorkerTask
from utils.buffer import SharedWorkerBuffer
from utils.timing import Timing
from utils.utils import (log, set_global_cuda_envvars, RWLock, assert_same_act_dim, get_shape_from_act_space)

# TODO: import other type of policies for other algorithms
from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy


class WorkerNode:
    def __init__(self, cfg, env_fn):
        # we should not use CUDA in the main thread, only on the workers
        set_global_cuda_envvars(cfg)
        self.cfg = cfg
        self.env_fn = env_fn

        self.worker_tasks = []

        self.transmitter = None

        self.local_ps = [None for _ in range(self.cfg.num_policies)]
        self.param_locks = [RWLock() for _ in range(self.cfg.num_policies)]
        self.ps_policy_versions = ((-1) * torch.ones(self.cfg.num_policies, dtype=torch.int64)).share_memory_()
        self.ps_ready_events = [mp.Event() for _ in range(self.cfg.num_policies)]

        self.task_finish_events = [mp.Event() for _ in range(self.cfg.num_tasks_per_node)]

        # ZeroMQ sockets to receive model parameters
        self._context = None
        self.model_weights_sockets = [None for _ in range(self.cfg.num_policies)]

        self.num_policy_updates = [0 for _ in range(self.cfg.num_policies)]
        self.model_weights_registries = [{} for _ in range(self.cfg.num_policies)]

    def init_shms(self, policy_id):
        num_agents = len(self.cfg.policy2agents[str(policy_id)])
        # shared memory allocation
        buffer = SharedWorkerBuffer(self.cfg, policy_id, num_agents, self.cfg.observation_space[policy_id],
                                    self.cfg.action_space[policy_id])

        envs_per_split = self.cfg.envs_per_actor // self.cfg.num_splits
        num_actors_per_task = self.cfg.num_actors // self.cfg.num_tasks_per_node
        num_actor_groups = num_actors_per_task // self.cfg.actor_group_size

        # initialize action/observation shared memories for communication between actors and policy workers
        # TODO: initialize env step outputs using config
        # following is just the case of StarCraft2 (policy-sharing environments)
        envstep_output_keys = [
            'observation_self', 'lidar', 'agent_qpos_qvel', 'box_obs', 'ramp_obs', 'mask_aa_obs', 'mask_ab_obs',
            'mask_ar_obs', 'mask_aa_obs_spoof', 'mask_ab_obs_spoof', 'mask_ar_obs_spoof', 'rewards',
            'fct_masks', 'rnn_states', 'rnn_states_critic'
        ]

        # actor workers consume actions and produce envstep_outputs in one shot (in env.step),
        # thus action_shms and envstep_output_semaphores only have one copy (not seperated for different policy_ids)
        envstep_output_semaphores = [[mp.Semaphore(0) for _ in range(self.cfg.num_splits)]
                                     for _ in range(num_actors_per_task)]

        assert_same_act_dim(self.cfg.action_space)
        act_dim = get_shape_from_act_space(self.cfg.action_space[0])
        act_shms = [[
            torch.zeros((envs_per_split, self.cfg.num_agents, act_dim), dtype=torch.int32).share_memory_().numpy()
            for _ in range(self.cfg.num_splits)
        ] for _ in range(num_actors_per_task)]

        # in the opposite side, different policy workers with different policy ids consume different subsets of
        # envstep_outputs and produce different subsets of actions asynchronously, thus act_semaphores and
        # envstep_output_shms have different copies for different policy ids
        act_semaphores = []
        envstep_output_shms = {}
        for k in envstep_output_keys:
            envstep_output_shms[k] = []
        envstep_output_shms['dones'] = []

        for controlled_agents in self.cfg.policy2agents.values():
            act_semaphores.append([[mp.Semaphore(0) for _ in range(self.cfg.num_splits)]
                                   for _ in range(num_actors_per_task)])

            num_agents = len(controlled_agents)

            for k in envstep_output_keys:
                if not hasattr(buffer, k):
                    continue

                shape = getattr(buffer, k).shape[4:]
                shape = (envs_per_split * self.cfg.actor_group_size, num_agents, *shape)

                envstep_output_shms[k].append([[
                    torch.zeros(shape, dtype=torch.float32).share_memory_().numpy() for _ in range(self.cfg.num_splits)
                ] for _ in range(num_actor_groups)])

            dones_shape = (envs_per_split * self.cfg.actor_group_size, num_agents, 1)
            envstep_output_shms['dones'].append([[
                torch.zeros(dones_shape, dtype=torch.float32).share_memory_().numpy()
                for _ in range(self.cfg.num_splits)
            ] for _ in range(num_actor_groups)])

        return buffer, act_shms, act_semaphores, envstep_output_shms, envstep_output_semaphores

    def init_sockets(self):
        self._context = zmq.Context()

        for i in range(self.cfg.num_policies):
            socket = self._context.socket(zmq.SUB)
            socket.connect(self.cfg.model_weights_addrs[i])
            socket.setsockopt(zmq.SUBSCRIBE, b'') # TODO, start str
            self.model_weights_sockets[i] = socket

    # receive msg
    def _update_weights(self, timing, policy_id, block=False):
        socket = self.model_weights_sockets[policy_id]
        model_weights_registry = self.model_weights_registries[policy_id]
        ps = self.local_ps[policy_id]
        lock = self.param_locks[policy_id]

        with timing.add_time('update_weights'), timing.time_avg('update_weights_once'):
            msg = None

            if block:
                msg = socket.recv_multipart(flags=0)
            else:
                while True:
                    # receive the latest model parameters
                    try:
                        msg = socket.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
            if msg is None:
                return

            # msg is multiple (key, tensor) pairs + policy version
            assert len(msg) % 2 == 1
            with lock.w_locked():
                for i in range(len(msg) // 2):
                    key, value = msg[2 * i].decode('ascii'), msg[2 * i + 1]

                    shape, dtype = model_weights_registry[key]
                    tensor = torch.from_numpy(np.frombuffer(memoryview(value), dtype=dtype).reshape(*shape))

                    ps[key][:] = tensor

                learner_policy_version = int(msg[-1].decode('ascii'))

                self.ps_policy_versions[policy_id] = learner_policy_version

        if self.num_policy_updates[policy_id] % 1 == 0:
            log.debug(
                'Updated Policy %d on node %d, policy_version %d (%s)',
                policy_id,
                self.cfg.worker_node_idx,
                learner_policy_version,
                str(timing.update_weights_once),
            )
        self.num_policy_updates[policy_id] += 1

    def _init(self, timing):
        with timing.add_time('init'):
            for policy_id in range(self.cfg.num_policies):
                example_agent = self.cfg.policy2agents[str(policy_id)][0]

                example_policy = Policy(torch.device('cpu'), self.cfg, self.cfg.observation_space[example_agent],
                                        self.cfg.action_space[example_agent], False)
                self.local_ps[policy_id] = {
                    k: v.detach().share_memory_()
                    for k, v in example_policy.state_dict().items()
                }

                for k, v in self.local_ps[policy_id].items():
                    self.model_weights_registries[policy_id][k] = (v.numpy().shape, v.numpy().dtype)

                del example_policy

            self.init_sockets()

            for task_rank in range(self.cfg.num_tasks_per_node):
                global_task_rank = self.cfg.worker_node_idx * self.cfg.num_tasks_per_node + task_rank
                policy_id = global_task_rank % self.cfg.num_policies

                (buffer, act_shms, act_semaphores, envstep_output_shms,
                 envstep_output_semaphores) = self.init_shms(policy_id)

                task = WorkerTask(self.cfg, task_rank, policy_id, self.env_fn, buffer, act_shms, act_semaphores,
                                  envstep_output_shms, envstep_output_semaphores, self.local_ps, self.param_locks,
                                  self.ps_policy_versions, self.ps_ready_events, self.task_finish_events[task_rank])
                task.start_process()
                self.worker_tasks.append(task)

            self.transmitter = Transmitter(self.cfg, [worker_task.buffer for worker_task in self.worker_tasks])
            self.transmitter.start_process()

        for policy_id in range(self.cfg.num_policies):
            self._update_weights(timing, policy_id, block=True)
            self.ps_ready_events[policy_id].set()

    def run(self):
        torch.multiprocessing.set_sharing_strategy('file_system')

        timing = Timing()

        self._init(timing)

        try:
            while not all([e.is_set() for e in self.task_finish_events]):
                for policy_id in range(self.cfg.num_policies):
                    self._update_weights(timing, policy_id)

                time.sleep(0.02)

        except Exception:
            log.exception('Exception in worker node loop')
        except KeyboardInterrupt:
            log.warning('Keyboard interrupt detected in worker node loop, exiting...')

        log.info('All tasks on worker node %d joined! Worker node mission accomplished.', self.cfg.worker_node_idx)
