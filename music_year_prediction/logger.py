from typing import Union

import git
import wandb
from omegaconf import DictConfig, OmegaConf

from .utils import flatten_json


class Logger:
    def __init__(self, cfg: DictConfig):
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.commit.hexsha
        wandb_conf = flatten_json(OmegaConf.to_container(cfg, resolve=True))
        wandb_conf["code"] = repo.git.rev_parse(sha, short=7)
        wandb.init(
            project="music_year_prediction",
            anonymous="must",
            config=wandb_conf,
            notes="A simple run from https://github.com/markblumenau/music_year_prediction",
        )

    def update(self, metric: Union[str, dict[str, str]] = None, value: float = 0.0):
        if type(metric) is dict:
            wandb.log(metric)
        else:
            wandb.log({metric: value})

    def finalize(self):
        wandb.finish()
