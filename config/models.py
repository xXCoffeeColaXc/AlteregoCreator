from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class DataConfig(BaseModel):
    root_dir: str
    image_dir: str
    attributes_file: str
    selected_attrs: Dict[int, Any]
    crop_size: int
    image_size: int


class ModelConfig(BaseModel):
    c_dim: int
    g_conv_dim: int
    d_conv_dim: int
    g_repeat_num: int
    d_repeat_num: int


class FolderConfig(BaseModel):
    output: str
    weights: str
    logs: str
    checkpoints: str
    samples: str


class TrainingConfig(BaseModel):
    device: str = 'cuda'
    batch_size: int  # Must be greater than 1
    random_seed: int
    epochs: int
    g_lr: float
    d_lr: float
    n_critics: int
    beta1: float
    beta2: float
    lambda_rec: float
    lambda_cls: float
    lambda_gp: float
    num_workers: int
    log_interval: int
    save_interval: int
    sample_interval: int
    resume_training: bool
    resume_checkpoint: Optional[str] = None
    sample_size: int
    num_grid_rows: int


class Config(BaseModel):
    training: TrainingConfig
    data: DataConfig
    model: ModelConfig
    folders: FolderConfig


# Example usage:
if __name__ == '__main__':
    import yaml

    def load_config(config_path: str) -> Config:
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return Config(**config_data)

    def dummy_function(ignore_index=None, reduction=None):
        print(f"ignore_index: {ignore_index}")
        print(f"reduction: {reduction}")

    config = load_config('config/config.yaml')
    print(config.training.epochs)
