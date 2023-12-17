import os
from datetime import datetime

from omegaconf import DictConfig, OmegaConf

from qcs import QCS_DIRECTORY_PATH


def main_decorator(func):
    def wrapper(cfg: DictConfig):
        # Extract the config
        cfg = OmegaConf.to_container(cfg, resolve=True)
        # Track whether the run has finished
        cfg["finished_running"] = False
        # Pre-run write
        run_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
        with open(
            os.path.join(
                QCS_DIRECTORY_PATH, cfg["write_config_directory"], f"{run_name}.yaml"
            ),
            "w",
        ) as f:
            OmegaConf.save(cfg, f)
        # Run the function
        func(run_name, cfg)
        # Post-run write
        cfg["finished_running"] = True
        with open(
            os.path.join(
                QCS_DIRECTORY_PATH, cfg["write_config_directory"], f"{run_name}.yaml"
            ),
            "w",
        ) as f:
            OmegaConf.save(cfg, f)

    return wrapper
