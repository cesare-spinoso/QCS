import os
from datetime import datetime

from omegaconf import DictConfig, OmegaConf


def main_decorator(func):
    def wrapper(cfg: DictConfig):
        # Extract the config
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg["finished_running"] = False
        # Pre-exp commit
        run_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
        with open(
            os.path.join(cfg["write_config_directory"], f"{run_name}.yaml"), "w"
        ) as f:
            OmegaConf.save(cfg, f)
        # Run the function
        func(run_name, cfg)

    return wrapper
