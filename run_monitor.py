import argparse
import yaml
from monitor.monitor import Monitor

def get_config():
    parser = argparse.ArgumentParser(description='monitor', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--is_head", action = "store_true")
    parser.add_argument("--name", type = str, default = "learner0")
    parser.add_argument("--config", type = str, default = "configs/starcraft2/config.yaml")
    
    return parser.parse_args()

if __name__ == '__main__':    
    cfg = get_config()
    if cfg.config is not None:
        with open(cfg.config) as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in cfg_dict.items():
            setattr(cfg, k, v)

    cfg.total_rounds = (cfg.train_for_seconds + 300) // cfg.interval
    # cfg.total_rounds = 50

    m = Monitor(cfg)
    m.start_monitor()
    m.run()

    # m.summary()
    m.terminate_monitor()
    m.close()
