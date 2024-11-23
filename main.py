import torch
from tqdm import tqdm
from losses import adverarial_loss, classification_loss, reconstruction_loss
from network import Generator, Discriminator
from dataloader import CelebA, get_loader
from torch.utils.data import DataLoader
from config.models import ModelConfig, TrainingConfig
from utils import load_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm - 1)**2)


def build_models(model_config: ModelConfig, train_config: TrainingConfig):
    G = Generator(model_config).to(device)
    D = Discriminator(model_config).to(device)

    g_optimizer = torch.optim.Adam(G.parameters(), train_config.g_lr,
                                   [train_config.beta1, train_config.beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), train_config.d_lr,
                                   [train_config.beta1, train_config.beta2])

    return G, D, g_optimizer, d_optimizer


def step_train_discriminator(D: Discriminator, G: Generator,
                             x_real: torch.Tensor, label_org: torch.Tensor,
                             c_trg: torch.Tensor,
                             config: TrainingConfig) -> torch.Tensor:
    # Compute loss with real images.
    out_src, out_cls = D(x_real)
    d_loss_real = adverarial_loss(out_src)
    d_loss_cls = classification_loss(out_cls, label_org)

    # Compute loss with fake images.
    x_fake = G(x_real, c_trg)
    out_src, out_cls = D(x_fake.detach())
    d_loss_fake = (-1) * adverarial_loss(out_src)

    # Compute loss for gradient penalty.
    alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)
    x_hat = (alpha * x_real.data +
             (1 - alpha) * x_fake.data).requires_grad_(True)
    out_src, _ = D(x_hat)
    d_loss_gp = gradient_penalty(out_src, x_hat)

    # Backward and optimize.
    d_loss = d_loss_real + d_loss_fake + config.lambda_cls * d_loss_cls + config.lambda_gp * d_loss_gp
    return d_loss


def step_train_generator(D: Discriminator, G: Generator, x_real: torch.Tensor,
                         label_trg: torch.Tensor, c_org: torch.Tensor,
                         c_trg: torch.Tensor,
                         config: TrainingConfig) -> torch.Tensor:

    # Original-to-target domain.
    x_fake = G(x_real, c_trg)
    out_src, out_cls = D(x_fake)
    g_loss_fake = adverarial_loss(out_src)
    g_loss_cls = classification_loss(out_cls, label_trg)

    # Target-to-original domain.
    x_reconst = G(x_fake, c_org)
    g_loss_rec = reconstruction_loss(x_real - x_reconst)

    # Backward and optimize.
    g_loss = g_loss_fake + config.lambda_rec * g_loss_rec + config.lambda_cls * g_loss_cls
    return g_loss


def train(G: Generator, D: Discriminator, g_optimizer: torch.optim.Optimizer,
          d_optimizer: torch.optim.Optimizer, dataloader: DataLoader,
          config: TrainingConfig):

    for epoch in range(config.epochs):

        pbar = tqdm(dataloader, leave=True)
        cur_itrs = 0

        for batch_idx, (x_real, label_org) in enumerate(pbar):
            cur_itrs += 1
            x_real = x_real.to(device)  # Input images.
            label_org = label_org.to(
                device)  # Labels for computing classification loss.

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(
                label_org.size(0))  # size: config.num_classes
            label_trg = label_org[rand_idx].to(
                device)  # Labels for computing classification loss.

            c_org = label_org.clone().to(device)  # Original domain labels.
            c_trg = label_trg.clone().to(device)  # Target domain labels.

            # Train discriminator
            d_loss = step_train_discriminator(D, G, x_real, label_org, c_trg,
                                              config)
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            if cur_itrs % config.n_critics == 0:
                g_loss = step_train_generator(D, G, x_real, label_trg, c_org,
                                              c_trg, config)
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                np_g_loss = g_loss.detach().cpu().numpy()

            np_d_loss = d_loss.detach().cpu().numpy()

            # Logging
            if cur_itrs % config.log_interval == 0:
                print(
                    f"Epoch [{epoch}/{config.epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], G_loss: {np_g_loss:.4f}, D_loss: {np_d_loss:.4f}"
                )

        # Save model checkpoints
        if epoch % config.save_interval == 0:
            pass

        # Decay learning rates


def validate():
    pass


def test():
    pass


if __name__ == '__main__':
    # Load configurations
    config = load_config('config/config.yaml')

    # Create data loader
    dataloader = get_loader(config.data, config.training, 'train')

    # Create models
    G, D, g_optimizer, d_optimizer = build_models(config.model)

    # Run training and validation
    train(G, D, g_optimizer, d_optimizer, dataloader, config.training)
