from pathlib import Path

from hydra import compose, initialize
from music_year_prediction import LinearModel, Logger, Trainer, make_datasets
from music_year_prediction.utils import get_optimizer
from omegaconf import DictConfig


def train(cfg: DictConfig):
    train_dataset, valid_dataset = make_datasets(cfg.dataset.get("train_size", 463715))
    # I know that it is better to have preprocessing such as standard scaler in dataset
    # But for the sake of inference working with raw data straight to the model
    # I do it this way
    # TODO: Make a separate Preprocesser
    model = LinearModel(
        block_count=cfg.model.get("block_count", 3),
        input_size=cfg.model.get("input_size", train_dataset.features_count),
        hidden_size=cfg.model.get("hidden_size", 1000),
        output_size=cfg.model.get("output_size", train_dataset.out),
        mean_features=train_dataset.mean_features,
        std_features=train_dataset.std_features,
        mean_target=train_dataset.mean_target,
        std_target=train_dataset.std_target,
    )
    logger = Logger(cfg)
    optimizer = get_optimizer(model.parameters(), cfg)

    save_name = Path(cfg.model.get("save_name", "./models/model.safetensors"))

    trainer = Trainer(
        model,
        optimizer,
        train_dataset,
        valid_dataset,
        logger,
        epochs=cfg.trainer.get("epochs", 5),
        batch_size=cfg.trainer.get("batch_size", 1024),
        save_name=save_name,
    )
    trainer.train()


def main():
    # Technically it is more than one function, but the logic here is obvious
    initialize(version_base=None, config_path="configs", job_name="music_year_prediction")
    cfg = compose(config_name="config")
    train(cfg)


if __name__ == "__main__":
    main()
