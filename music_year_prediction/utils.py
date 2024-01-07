from omegaconf import DictConfig
from torch import optim


# Because WandB has issues
# See https://github.com/wandb/wandb/issues/982


def flatten_json(json_like):
    json_out = json_like
    if type(json_out) is dict:
        for key, value in list(json_out.items()):
            if type(value) is dict:
                flatten_json(value)
                json_out.pop(key)
                for key_inner, value_inner in value.items():
                    json_out[key + "." + key_inner] = value_inner
    return json_out


def get_optimizer(params, cfg: DictConfig):
    if cfg.optim.get("optim_type", "Adam") == "Adam":
        return optim.Adam(params, lr=cfg.optim.get("lr", 1e-3))
    else:
        return optim.SGD(params, lr=cfg.optim.get("lr", 1e-2))
