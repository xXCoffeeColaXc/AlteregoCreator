from config.models import Config
import yaml
import torch
from itertools import product
import random
import os
import wandb


def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return Config(**config_data)


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def setup_logger(config: Config, device: torch.device):
    wandb.init(
        project='alterego-stargan',
        config={
            "image_size": config.data.image_size,
            "batch_size": config.training.batch_size,
            "epochs": config.training.epochs,
            "d_lr": config.training.d_lr,
            "g_lr": config.training.g_lr,
            "lambda_cls": config.training.lambda_cls,
            "lambda_rec": config.training.lambda_rec,
            "lambda_gp": config.training.lambda_gp,
            "n_critics": config.training.n_critics,
            "random_seed": config.training.random_seed,
        }
    )
    # Ensure DEVICE is tracked in WandB
    wandb.config.update({"device": device})


def create_run():
    checkpoint_path = 'outputs/checkpoints'
    sample_path = 'outputs/samples'
    max_run_id = _find_max_run_id(checkpoint_path)

    new_run_id = max_run_id + 1
    run_dir = f'run_{new_run_id}'
    new_sample_path = os.path.join(sample_path, run_dir)
    new_checkpoint_path = os.path.join(checkpoint_path, run_dir)

    # Create new directories for checkpoints
    os.makedirs(new_sample_path, exist_ok=True)
    os.makedirs(new_checkpoint_path, exist_ok=True)

    print(f"New run directory created: {run_dir}")

    return run_dir


def _find_max_run_id(checkpoint_path):
    max_run_id = 0
    if os.path.exists(checkpoint_path):
        for dir_name in os.listdir(checkpoint_path):
            if dir_name.startswith('run_'):
                try:
                    run_id = int(dir_name.split('_')[1])
                    max_run_id = max(max_run_id, run_id)
                except ValueError:
                    pass  # Ignore directories with non-integer run numbers

    return max_run_id


# Generate one valid permutation for each sample in the batch
# (B, num_attr) -> (B, num_attr)
def generate_one_valid_permutation_batch(batch_tensor: torch.Tensor, selected: list) -> torch.Tensor:

    age_attributes = ['Old', 'Young']
    gender_attributes = ['Male', 'Female']
    hair_attributes = ['Black_Hair', 'Blond_Hair', 'Brown_Hair']
    attr_to_index = {attr: idx for idx, attr in enumerate(selected)}
    batch_size = batch_tensor.size(0)
    num_attributes = len(selected)

    perm_batch = torch.zeros((batch_size, num_attributes), dtype=batch_tensor.dtype, device=batch_tensor.device)

    def select_attributes(group_attributes):
        num_options = len(group_attributes)
        chosen_indices = torch.randint(
            low=0, high=num_options, size=(batch_size,)
        )  # Randomly choose indices: shape (batch_size,)
        selected_attrs = [group_attributes[idx] for idx in chosen_indices.tolist()]
        return selected_attrs

    # Select one attribute from each group for all samples
    chosen_ages = select_attributes(age_attributes)
    chosen_genders = select_attributes(gender_attributes)
    chosen_hairs = select_attributes(hair_attributes)

    # Set the chosen age attributes
    age_indices = [attr_to_index[age] for age in chosen_ages]
    perm_batch[torch.arange(batch_size), age_indices] = 1.0

    # Set the chosen gender attributes
    gender_indices = [attr_to_index[gender] for gender in chosen_genders]
    perm_batch[torch.arange(batch_size), gender_indices] = 1.0

    # Set the chosen hair color attributes
    hair_indices = [attr_to_index[hair] for hair in chosen_hairs]
    perm_batch[torch.arange(batch_size), hair_indices] = 1.0

    return perm_batch


def generate_valid_permutations(input_tensor: torch.Tensor, selected: list) -> list[torch.Tensor]:
    age_attributes = ['Old', 'Young']
    gender_attributes = ['Male', 'Female']
    hair_attributes = ['Black_Hair', 'Blond_Hair', 'Brown_Hair']
    attr_to_index = {attr: idx for idx, attr in enumerate(selected)}

    valid_permutations = []

    # Iterate over all possible combinations within each group
    for age, gender, hair in product(age_attributes, gender_attributes, hair_attributes):
        perm = torch.zeros_like(input_tensor)

        # Set the age attribute
        if age == 'Old':
            perm[attr_to_index['Old']] = 1.0
            perm[attr_to_index['Young']] = 0.0
        else:
            perm[attr_to_index['Young']] = 1.0
            perm[attr_to_index['Old']] = 0.0

        # Set the gender attribute
        if gender == 'Male':
            perm[attr_to_index['Male']] = 1.0
            perm[attr_to_index['Female']] = 0.0
        else:
            perm[attr_to_index['Female']] = 1.0
            perm[attr_to_index['Male']] = 0.0

        # Set the hair color attribute
        perm[attr_to_index[hair]] = 1.0
        # Ensure other hair colors are set to 0
        for other_hair in hair_attributes:
            if other_hair != hair:
                perm[attr_to_index[other_hair]] = 0.0

        valid_permutations.append(perm)

    return valid_permutations
