#!/usr/bin/env python3
"""Custom script to compute normalization stats for your model."""

import pathlib
import sys
import tqdm
import tyro

import numpy as np

# Add the src directory to the path
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from openpi.shared import normalize
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.models import model as _model


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """Create a PyTorch data loader."""
    from openpi.training.data_loader import create_torch_dataloader as _create_torch_dataloader
    return _create_torch_dataloader(
        data_config, action_horizon, batch_size, model_config, num_workers, max_frames
    )


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """Create an RLDS data loader."""
    from openpi.training.data_loader import create_rlds_dataloader as _create_rlds_dataloader
    return _create_rlds_dataloader(
        data_config, action_horizon, batch_size, max_frames
    )


def main(config_name: str, output_dir: str, max_frames: int | None = None):
    """Compute normalization stats and save to custom directory."""
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # Use custom output directory
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
