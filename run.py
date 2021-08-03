import argparse
# from monitor import Monitor
import time
import yaml
import os
import subprocess
from monitor.script import script, to_bash_file, to_stdout

# specific config for different nodes
def get_config():
    parser = argparse.ArgumentParser(description='monitor', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--is_head", action = "store_true")
    parser.add_argument("--name", type = str, default = "learner0")
    parser.add_argument("--config", type = str, default = "configs/starcraft2/config.yaml")
    
    return parser.parse_args()

def parse_config(cfg):
    cwd = cfg.cwd
    cfg.trainer_dir = os.path.join(cwd, cfg.trainer_dir)
    cfg.worker_dir = os.path.join(cwd, cfg.worker_dir)
    cfg.monitor_dir = os.path.join(cwd, cfg.monitor_dir)
    cfg.log_dir = os.path.join(cwd, cfg.log_dir)
    cfg.config_dir = os.path.join(cwd, cfg.config)
    cfg.trainer_logdir = os.path.join(cfg.log_dir, cfg.trainer_logname)
    cfg.worker_logdir = os.path.join(cfg.log_dir, cfg.worker_logname)
    cfg.monitor_logdir = os.path.join(cfg.log_dir, cfg.monitor_logname)
    return cfg

if __name__ == "__main__":
    cfg = get_config()
    if cfg.config is not None:
        with open(cfg.config) as f:
            cfg_dict = yaml.load(f, Loader=yaml.Loader)
        for k, v in cfg_dict.items():
            setattr(cfg, k, v)
    
    cfg = parse_config(cfg)
    generated_script_dir =  "monitor/generated/launch.sh"

    # to_stdout(script(cfg, run_trainer = True, run_worker = True, run_monitor = True))
    to_bash_file(script(cfg, run_trainer = True, run_worker = True, run_monitor = True), scriptdir = generated_script_dir)

    # run scripts
    
    # works for python2, in python3 use subprocess.run()
    # rc = subprocess.call(["bash", "-c", "./update.sh"])
    # if not rc == 0:
    #     print("Update return non-zero code.")
    # time.sleep(0.5)
    
    try:
        out = subprocess.check_output(["bash", "-c", generated_script_dir])
        print(out)
    except subprocess.CalledProcessError as e:
        print(e)
