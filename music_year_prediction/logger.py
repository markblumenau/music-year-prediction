import warnings
from typing import Union

import git
import mlflow
import wandb
from omegaconf import DictConfig, OmegaConf

from .utils import flatten_json


class Logger:
    def __init__(self, cfg: DictConfig):
        self.logger_type = cfg.logger.get("type", "wandb")

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.commit.hexsha
        conf = flatten_json(OmegaConf.to_container(cfg, resolve=True))
        conf["commit_id"] = repo.git.rev_parse(sha, short=7)
        if self.logger_type == "wandb":
            wandb.init(
                project="music_year_prediction",
                anonymous="must",
                config=conf,
                notes="A simple run from https://github.com/markblumenau/music_year_prediction",
            )
        elif self.logger_type == "mlflow":
            # MLFlow requires a step value
            # I make it so Logger is responsible for it
            self.train_step = 0
            self.val_step = 0
            mlflow.set_tracking_uri(cfg.logger.get("mlflow_uri", "128.0.1.1:8080"))
            mlflow.log_params(conf)
        else:
            warnings.warn(
                "You specified an unknown logger type. No logger will be used.",
                stacklevel=2,
            )

    def update(
        self,
        metric: Union[str, dict[str, str]] = None,
        value: float = 0.0,
        on: str = "train",
    ):
        if type(metric) is str:
            metric = {metric: value}

        if self.logger_type == "wandb":
            wandb.log(metric)
        elif self.logger_type == "mlflow":
            if on == "train":
                mlflow.log_metrics(metric, step=self.train_step)
                self.train_step += 1
            elif on == "val":
                mlflow.log_metrics(metric, step=self.val_step)
                self.val_step += 1

    def finalize(self):
        if self.logger_type == "wandb":
            wandb.finish()
        elif self.logger_type == "mlflow":
            mlflow.end_run()
