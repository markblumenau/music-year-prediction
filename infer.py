import os
from pathlib import Path

import pandas as pd
from hydra import compose, initialize
from music_year_prediction import LinearModel
from omegaconf import DictConfig


def infer(cfg: DictConfig):
    model = LinearModel(
        block_count=cfg.model.get("block_count", 3),
        input_size=cfg.model.get("input_size", 90),
        hidden_size=cfg.model.get("hidden_size", 1000),
        output_size=cfg.model.get("output_size", 1),
    )
    save_name = Path(cfg.model.get("save_name", "./models/model.safetensors"))
    model.load(pull_dvc=cfg.model.get("pull_dvc", False), load_name=save_name)
    input = None
    if cfg.model.get("predict_on", None) is not None:
        input = Path(cfg.model.get("predict_on", None))

    result = pd.DataFrame({"prediction": model.predict(input=input)})
    os.makedirs("results", exist_ok=True)
    result.to_csv("results/test_predictions.csv")


def main():
    initialize(version_base=None, config_path="configs", job_name="music_year_prediction")
    cfg = compose(config_name="config")
    infer(cfg)


if __name__ == "__main__":
    main()
