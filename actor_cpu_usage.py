import psutil
import pgrep
import numpy as np
import time

if __name__ == "__main__":
    smac_pids = pgrep.pgrep('Main_Thread')
    actor_pids = pgrep.pgrep('python3.8')
    smac_procs = []
    for pid in smac_pids:
        smac_procs.append(psutil.Process(pid))
    actor_procs = []
    for pid in actor_pids:
        actor_procs.append(psutil.Process(pid))

    while True:
        time.sleep(2)
        smac_cpu_usage = []
        for p in smac_procs:
            smac_cpu_usage.append(p.cpu_percent())

        actor_cpu_usage = []
        for p in actor_procs:
            actor_cpu_usage.append(p.cpu_percent())
        # assert len(smac_cpu_usage) % len(actor_cpu_usage) == 0, (len(smac_cpu_usage), len(actor_cpu_usage))
        print('-' * 20)
        env_per_split = len(smac_cpu_usage) // (len(actor_cpu_usage) - 1)
        print(env_per_split)
        print("{} Actor Processes Consume {:.2f} CPUs in Total, Average {:.2f} CPU/Actor".format(
            len(actor_cpu_usage), sum(actor_cpu_usage) / 100, np.mean(actor_cpu_usage) / 100))
        print("{} SMAC game Processes Consume {:.2f} CPUs in Total, Average {:.2f} CPU/Actor".format(
            len(smac_cpu_usage), sum(smac_cpu_usage) / 100, np.mean(smac_cpu_usage) / 100))
        print("Rollout CPU Consumption {:.2f} CPUs/Actor".format(
            np.mean(actor_cpu_usage) / 100 + env_per_split * np.mean(smac_cpu_usage) / 100))
        print('-' * 20)
