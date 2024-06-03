from pathlib import Path
from typing import Any, cast
from hydra import compose, initialize
from omegaconf import OmegaConf

def compose_config(
    overrides: list[str] | None = None,
    config_path: Path = Path("conf"),
    config_name: str = "config",
    ) -> dict:

    with initialize(version_base=None, config_path=str(config_path)):
        hydra_config = compose(config_name=config_name, overrides=overrides)
        dict_config: dict[str, Any] = cast(
            dict[str, Any], OmegaConf.to_container(hydra_config, resolve=True)
        )
    return dict_config