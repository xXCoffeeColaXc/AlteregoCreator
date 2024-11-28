import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import LambdaLR
from losses import adverarial_loss, classification_loss, reconstruction_loss
from network import Generator, Discriminator
from dataloader import CelebA, get_loader, get_transform
from torch.utils.data import DataLoader
from config.models import ModelConfig, TrainingConfig, FolderConfig, Config
from utils import load_config, denorm, generate_valid_permutations, generate_one_valid_permutation_batch, create_run, setup_logger
from torchvision.utils import save_image
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
run_id = create_run()


def get_linear_decay_scheduler(optimizer: torch.optim.Optimizer, num_epochs: int, num_epochs_decay: int):

    def lambda_lr(epoch):
        if epoch < (num_epochs - num_epochs_decay):
            return 1.0
        else:
            return max(0.0, 1.0 - float(epoch - (num_epochs - num_epochs_decay)) / num_epochs_decay)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)
    return scheduler


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates, _ = D(interpolates)

    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty


def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(
        outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True
    )[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm - 1)**2)


def build_models(model_config: ModelConfig, train_config: TrainingConfig):
    G = Generator(model_config).to(device)
    D = Discriminator(model_config).to(device)

    g_optimizer = torch.optim.Adam(G.parameters(), train_config.g_lr, [train_config.beta1, train_config.beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), train_config.d_lr, [train_config.beta1, train_config.beta2])

    return G, D, g_optimizer, d_optimizer


def load_checkpoint(checkpoint_path: Path, model_config: ModelConfig):
    G = Generator(model_config).to(device)
    checkpoint = torch.load(checkpoint_path)
    G.load_state_dict(checkpoint['G'])
    return G


def save_checkpoint(
    G: Generator,
    D: Discriminator,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    epoch: int,
    checkpoint_dir: Path
):
    G_path = checkpoint_dir / str(run_id) / f'{epoch}-G.ckpt'
    D_path = checkpoint_dir / str(run_id) / f'{epoch}-D.ckpt'

    torch.save({'G': G.state_dict(), 'g_optimizer': g_optimizer.state_dict(), 'epoch': epoch}, G_path)
    torch.save({'D': D.state_dict(), 'd_optimizer': d_optimizer.state_dict(), 'epoch': epoch}, D_path)

    print(f"Checkpoint saved at {G_path} and {D_path}")


def calculate_discriminator_loss(
    D: Discriminator,
    G: Generator,
    x_real: torch.Tensor,
    label_org: torch.Tensor,
    c_trg: torch.Tensor,
    config: TrainingConfig
) -> torch.Tensor:
    # Compute loss with real images.
    out_src, out_cls = D(x_real)
    d_loss_real = -torch.mean(out_src)
    d_loss_cls = classification_loss(out_cls, label_org)

    # Compute loss with fake images.
    x_fake = G(x_real, c_trg)
    out_src, out_cls = D(x_fake.detach())
    d_loss_fake = torch.mean(out_src)

    # Compute loss for gradient penalty.
    alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)
    x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
    out_src, _ = D(x_hat)
    d_loss_gp = gradient_penalty(out_src, x_hat)
    #d_loss_gp2 = compute_gradient_penalty(D, x_real, x_fake)
    #print(f"d_loss_gp new: {d_loss_gp2}")

    # Wasserstein loss
    # L_D = E[D_src(x)] - E[D_src(G(z))]
    #d_wasserstein = d_loss_real + d_loss_fake - config.lambda_gp * d_loss_gp

    # Backward and optimize.
    #d_loss = -d_wasserstein + config.lambda_cls * d_loss_cls
    d_loss = d_loss_real + d_loss_fake + config.lambda_cls * d_loss_cls + config.lambda_gp * d_loss_gp
    return d_loss


def calculate_generator_loss(
    D: Discriminator,
    G: Generator,
    x_real: torch.Tensor,
    label_trg: torch.Tensor,
    c_org: torch.Tensor,
    c_trg: torch.Tensor,
    config: TrainingConfig
) -> torch.Tensor:

    # Original-to-target domain.
    x_fake = G(x_real, c_trg)
    out_src, out_cls = D(x_fake)
    g_loss_fake = -torch.mean(out_src)
    g_loss_cls = classification_loss(out_cls, label_trg)

    # Target-to-original domain.
    x_reconst = G(x_fake, c_org)
    g_loss_rec = reconstruction_loss(x_real, x_reconst)

    # Backward and optimize.
    g_loss = g_loss_fake + config.lambda_rec * g_loss_rec + config.lambda_cls * g_loss_cls
    return g_loss


def train(
    G: Generator,
    D: Discriminator,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    g_sheduler: torch.optim.lr_scheduler.LinearLR,
    d_sheduler: torch.optim.lr_scheduler.LinearLR,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    config: Config
):
    train_config = config.training
    data_config = config.data

    for epoch in range(1, train_config.epochs + 1):

        pbar = tqdm(train_dataloader, desc=f"Epoch [{epoch}/{train_config.epochs}]", leave=True)
        cur_itrs = 0
        d_loss_epoch = 0.0
        g_loss_epoch = 0.0

        for batch_idx, (x_real, label_org) in enumerate(pbar):
            cur_itrs += 1
            x_real = x_real.to(device)  # Input images.
            label_org = label_org.to(device)  # Labels for computing classification loss.

            # Generate target domain labels randomly.
            label_trg = generate_one_valid_permutation_batch(label_org, data_config.selected_attrs).to(device)

            c_org = label_org.clone().to(device)  # Original domain labels.
            c_trg = label_trg.clone().to(device)  # Target domain labels.

            # Train discriminator
            d_loss = calculate_discriminator_loss(D, G, x_real, label_org, c_trg, train_config)
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            d_loss_epoch += d_loss.item()

            # Train generator
            if cur_itrs % train_config.n_critics == 0:
                g_loss = calculate_generator_loss(D, G, x_real, label_trg, c_org, c_trg, train_config)
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                g_loss_epoch += g_loss.item()

                # Update progress bar with generator loss
                pbar.set_postfix(
                    {
                        "G_loss": f"{g_loss.item():.4f}",
                        "D_loss": f"{d_loss.item():.4f}",
                        "G_lr": f"{g_optimizer.param_groups[0]['lr']:.6f}",
                        "D_lr": f"{d_optimizer.param_groups[0]['lr']:.6f}"
                    }
                )

                wandb.log({"G_loss": g_loss.item(), "D_loss": d_loss.item()})

        # Step learning rate scheduler
        g_sheduler.step()
        d_sheduler.step()

        # Calculate average losses for the epoch
        avg_d_loss = d_loss_epoch / len(train_dataloader)
        avg_g_loss = g_loss_epoch / (len(train_dataloader) // train_config.n_critics)

        # Logging at the end of the epoch
        print(
            f"Epoch [{epoch}/{train_config.epochs}] \t Avg_G_loss: {avg_g_loss:.4f} \t Avg_D_loss: {avg_d_loss:.4f} \t G_lr: {g_optimizer.param_groups[0]['lr']:.6f} \t D_lr: {d_optimizer.param_groups[0]['lr']:.6f}"
        )

        wandb.log(
            {
                "Epoch": epoch,
                "Avg_G_loss": avg_g_loss,
                "Avg_D_loss": avg_d_loss,
                "G_lr": g_optimizer.param_groups[0]['lr'],
                "D_lr": d_optimizer.param_groups[0]['lr']
            }
        )

        # Validate model
        if epoch % train_config.val_interval == 0:
            validate(G, D, val_dataloader, train_config, data_config.selected_attrs)

        # Save model checkpoints
        if epoch % train_config.save_interval == 0:
            save_dir = Path(config.folders.checkpoints)
            save_checkpoint(G, D, g_optimizer, d_optimizer, epoch, save_dir)


def validate(
    G: Generator, D: Discriminator, val_dataloader: DataLoader, train_config: TrainingConfig, selected_attrs: list
):
    G.eval()
    D.eval()
    d_losses = []
    g_losses = []

    with torch.no_grad():
        for batch_idx, (x_real, label_org) in enumerate(val_dataloader):
            x_real = x_real.to(device)  # Input images.
            label_org = label_org.to(device)  # Labels for computing classification loss.

            # Generate target domain labels randomly.
            label_trg = generate_one_valid_permutation_batch(label_org, selected_attrs).to(device)

            c_org = label_org.clone().to(device)  # Original domain labels.
            c_trg = label_trg.clone().to(device)  # Target domain labels.

            # Calculate discriminator loss
            out_src, out_cls = D(x_real)
            d_loss_real = -torch.mean(out_src)
            d_loss_cls = classification_loss(out_cls, label_org)
            # Compute loss with fake images.
            x_fake = G(x_real, c_trg)
            out_src, out_cls = D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)
            d_loss = d_loss_real + d_loss_fake + train_config.lambda_cls * d_loss_cls

            # Calculate generator loss
            # Original-to-target domain.
            x_fake = G(x_real, c_trg)
            out_src, out_cls = D(x_fake)
            g_loss_fake = -torch.mean(out_src)
            g_loss_cls = classification_loss(out_cls, label_trg)

            # Target-to-original domain.
            x_reconst = G(x_fake, c_org)
            g_loss_rec = reconstruction_loss(x_real, x_reconst)
            g_loss = g_loss_fake + train_config.lambda_rec * g_loss_rec + train_config.lambda_cls * g_loss_cls

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

    D.train()
    G.train()

    print(f"Validation: G_loss: {np.mean(g_losses):.4f} \t D_loss: {np.mean(d_losses):.4f}")
    wandb.log({"Val_G_loss": np.mean(g_losses), "Val_D_loss": np.mean(d_losses)})


def test(G: Generator, test_dataloader: DataLoader, config: Config):
    with torch.no_grad():
        for i, (x_real, c_org) in enumerate(test_dataloader):

            # Prepare input images and target domain labels.
            x_real_singe = x_real[0].unsqueeze(0).to(device)
            c_org_sinle = c_org[0]
            c_trg_list = generate_valid_permutations(c_org_sinle, config.data.selected_attrs)  # not a batch

            # Translate images.
            x_fake_list = [x_real_singe]
            for c_trg in c_trg_list:
                c_trg_single = c_trg.unsqueeze(0).to(device)
                x_fake: torch.Tensor = G(x_real_singe, c_trg_single)
                x_fake_list.append(x_fake)

            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = Path(config.folders.samples) / str(run_id) / f'{i+1}-images.jpg'
            save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            print('Saved real and fake images into {}...'.format(result_path))


def infer_and_save(G: Generator, image, c_org: torch.Tensor, config: Config):
    with torch.no_grad():
        # Load and preprocess the image
        # transform = get_transform(config.data, 'test')
        # image = Image.open(image_path).convert('RGB')
        # x_real = transform(image).unsqueeze(0).to(device)
        x_real_singe = image[0].unsqueeze(0).to(device)
        c_org_sinle = c_org[0]
        print(f"{c_org_sinle}")
        print(f"{c_org_sinle.shape}")
        print(f"{x_real_singe.shape}")
        # Generate target labels permutations
        c_trg_list = generate_valid_permutations(c_org_sinle, config.data.selected_attrs)

        # Generate images
        x_fake_list = [x_real_singe]
        for idx, c_trg in enumerate(c_trg_list):
            c_trg_single = c_trg.unsqueeze(0).to(device)

            x_fake: torch.Tensor = G(x_real_singe, c_trg_single)
            x_fake_list.append(x_fake)

        # Concatenate images along width (dim=3)
        x_concat = torch.cat(x_fake_list, dim=3)

        # Save the concatenated images
        result_path = Path('test') / 'inference_result.jpg'
        save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
        print(f'Saved real and fake images into {result_path}...')


if __name__ == '__main__':
    # Load configurations
    config = load_config('config/config.yaml')

    # #Setup logger
    setup_logger(config, device)

    # Create data loader
    train_dataloader = get_loader(config.data, config.training, 'train')
    val_dataloader = get_loader(config.data, config.training, 'val')

    # Create models
    G, D, g_optimizer, d_optimizer = build_models(config.model, config.training)
    g_sheduler = get_linear_decay_scheduler(g_optimizer, config.training.epochs, config.training.epochs // 2)
    d_sheduler = get_linear_decay_scheduler(d_optimizer, config.training.epochs, config.training.epochs // 2)

    # Run training and validation
    train(
        G=G,
        D=D,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        g_sheduler=g_sheduler,
        d_sheduler=d_sheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
    )

    wandb.finish()

    # test_dataloader = get_loader(config.data, config.training, 'test')
    # test_image, label = next(iter(test_dataloader))
    # print(f"{test_image.shape}")
    # print(f"{label.shape}")
    # print(type(test_image))
    # print(type(label))
    # # G = load_checkpoint(Path(config.folders.checkpoints) / '30-G.ckpt', config.model)
    # # test(G, test_dataloader, config)

    # # Load config and model
    # G = load_checkpoint(Path(config.folders.checkpoints) / '30-G.ckpt', config.model)
    # G.to(device)
    # G.eval()

    # # Prepare the original label tensor

    # # Run inference and save images
    # infer_and_save(G, test_image, label, config)
