import torch
import torch.nn as nn
from tqdm import tqdm

from data.visualize import show

# Generator

def genBlock(inp_nodes, out_nodes):
  return nn.Sequential(
      nn.Linear(inp_nodes, out_nodes),
      nn.BatchNorm1d(out_nodes),
      nn.ReLU()
  )

def gen_noise(batch_size, latent_dim, device="cuda"):
  return torch.randn(batch_size, latent_dim).to(device)


class Generator(nn.Module):
  def __init__(self,latent_dim=64, out_dim=784, hidden_dim=120):
    super().__init__()

    self.latent_dim = latent_dim
    self.out_dim = out_dim
    self.hidden_dim = hidden_dim

    self.gen = nn.Sequential(
        genBlock(latent_dim, hidden_dim), # 64, 128
        genBlock(hidden_dim, hidden_dim*2), # 128 , 256
        genBlock(hidden_dim*2, hidden_dim*4), # 256, 512
        genBlock(hidden_dim*4, hidden_dim*8), # 512, 1024
        genBlock(hidden_dim*8, out_dim), # 1024, 784 (28*28)
        nn.Sigmoid(),
    )

  def forward(self, noise):
    return self.gen(noise)
  

def gen_loss(loss_func, gen, disc, batch_size, z_dim):
  noise = gen_noise(batch_size, z_dim)
  fake = gen(noise)
  pred = disc(fake)
  target = torch.ones_like(pred)
  gen_loss = loss_func(pred, target)

  return gen_loss
     

# Discriminator

def discBlock(inp_nodes, out_nodes, act=nn.LeakyReLU(0.2)):
  return nn.Sequential(
      nn.Linear(inp_nodes, out_nodes),
      act,
  )


class Discriminator(nn.Module):

  def __init__(self,inp_dim = 784, hidden_dim=256):
    super().__init__()

    self.inp_dim = inp_dim
    self.hidden_dim = hidden_dim

    self.disc = nn.Sequential(
        discBlock(inp_dim, hidden_dim*4),
        discBlock(hidden_dim*4, hidden_dim*2),
        discBlock(hidden_dim*2, hidden_dim),
        nn.Linear(hidden_dim, 1)

    )

  def forward(self,image):
    return self.disc(image)


def disc_loss(loss_func, gen, disc, batch_size, z_dim, real):
  noise = gen_noise(batch_size, z_dim)
  fake = gen(noise)
  disc_fake = disc(fake.detach())
  disc_fake_target = torch.zeros_like(disc_fake)
  disc_fake_loss = loss_func(disc_fake, disc_fake_target)

  disc_real = disc(real)
  disc_real_target = torch.ones_like(disc_real)
  disc_real_loss = loss_func(disc_real, disc_real_target)

  disc_loss = (disc_fake_loss + disc_real_loss)/2

  return disc_loss
     

def train_gan(gen, disc, dataloader, gen_opt, disc_opt, loss_func, z_dim, epochs, device, info_iter=100):
    """
    Trains a basic GAN model.

    Parameters:
        gen (nn.Module): The generator model.
        disc (nn.Module): The discriminator model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the real data.
        gen_opt (torch.optim.Optimizer): Optimizer for the generator.
        disc_opt (torch.optim.Optimizer): Optimizer for the discriminator.
        loss_func: The loss function to use (e.g., BCE Loss).
        z_dim (int): Dimensionality of the latent space.
        epochs (int): Number of epochs for training.
        device (str): Device to train on ("cuda" or "cpu").
        info_iter (int): Iteration interval for displaying stats and visualizations.

    Returns:
        dict: Contains lists of losses and iteration steps for visualization.
    """

    cur_iter = 0
    mean_gen_loss = 0
    mean_disc_loss = 0
    mean_disc_loss_list = []
    mean_gen_loss_list = []
    iters_list = []

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for real_image in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
            # Discriminator Step
            disc_opt.zero_grad()  # Reset gradients

            cur_batch_size = len(real_image)
            real_image = real_image.view(cur_batch_size, -1).to(device)  # Reshape and move to device

            # Compute discriminator loss
            disc_losses = disc_loss(loss_func, gen, disc, cur_batch_size, z_dim, real_image)
            disc_losses.backward()  # Compute gradients
            disc_opt.step()  # Update discriminator weights

            # Generator Step
            gen_opt.zero_grad()  # Reset gradients

            # Compute generator loss
            gen_losses = gen_loss(loss_func, gen, disc, cur_batch_size, z_dim)
            gen_losses.backward()  # Compute gradients
            gen_opt.step()  # Update generator weights

            # Compute and store stats
            mean_disc_loss += disc_losses.item() / info_iter
            mean_gen_loss += gen_losses.item() / info_iter

            # Visualization and stats logging
            if cur_iter % info_iter == 0 and cur_iter > 0:
                fake_noise = gen_noise(cur_batch_size, z_dim, device)
                fake = gen(fake_noise)
                #fake gives the logits of each pixel, we now need to convert it to 0/1 ?
                
                print(f"shape of fake: {fake.shape}")
                print(f"shape of real_image: {real_image.shape}")
                print(f" for the first fae image the sum of values is {fake[0].sum()}")
                # Display real and fake images
                print(f"Step {cur_iter}, Generator Loss: {mean_gen_loss:.4f}, Discriminator Loss: {mean_disc_loss:.4f}")
                show(real_image[:16].cpu())  # Show first 16 real images
                show(fake[:16].detach().cpu())  # Show first 16 fake images
                
                # Reset mean losses
                mean_gen_loss_list.append(mean_gen_loss)
                mean_disc_loss_list.append(mean_disc_loss)
                mean_gen_loss, mean_disc_loss = 0, 0

            iters_list.append(cur_iter)
            cur_iter += 1

    return {
        "mean_gen_loss": mean_gen_loss_list,
        "mean_disc_loss": mean_disc_loss_list,
        "iterations": iters_list
    }
     
