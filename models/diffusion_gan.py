import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import numpy as np

from ddpm import get_beta_schedule, q_sample, extract
from ddpm_unet import TimeEmbedding, UNet
from gan import discBlock


def get_time_schedule(timesteps, device):
    eps_small = 1e-3
    t = np.arange(0, timesteps + 1, dtype=np.float64)
    t = t / timesteps
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

class Diffusion_Coefficients():
    def __init__(self, timesteps, beta_start=1e-4, beta_end=0.02, device: torch.device = "cuda"):
                
        betas = get_beta_schedule(timesteps, beta_start, beta_end)
        self.sigmas = betas**0.5
        self.a_s = torch.sqrt(1-betas)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)
    
class Posterior_Coefficients():
    def __init__(self, timesteps, beta_start=1e-4, beta_end=0.02, device: torch.device = "cuda"):
        
        self.betas = get_beta_schedule(timesteps, beta_start, beta_end)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(x_start, t, coeff.a_s_cum, coeff.sigmas_cum)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    return x_t, x_t_plus_one

def sample_posterior(coefficients, x_0,x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos

def sample_from_model(coefficients, generator, n_time, x_init):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            x_0 = generator(x, t_time)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
        
    return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, downsample=False, act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.downsample = downsample
        self.act = act
        self.time_proj = nn.Linear(t_emb_dim, out_channels)
        self.main = discBlock(in_channels, out_channels)
        if downsample:
            self.downsample_layer = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t_emb):
        t_proj = self.time_proj(t_emb)[:, :, None, None]
        x = self.main(x + t_proj)
        if self.downsample:
            x = self.downsample_layer(x)
        return x
    
class DiscriminatorTimeDependent(nn.Module):
    def __init__(self, nc=3, ngf=64, t_emb_dim=128, act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.act = act
        self.t_embed = TimeEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )
        self.start_conv = nn.Conv2d(nc * 2, ngf * 2, kernel_size=1, padding=0)  # Concatenated x and x_t
        self.conv1 = DownBlock(ngf * 2, ngf * 2, t_emb_dim=t_emb_dim, act=act)
        self.conv2 = DownBlock(ngf * 2, ngf * 4, t_emb_dim=t_emb_dim, downsample=True, act=act)
        self.conv3 = DownBlock(ngf * 4, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)
        self.conv4 = DownBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)
        self.final_conv = nn.Conv2d(ngf * 8 + 1, ngf * 8, kernel_size=3, padding=1)
        self.end_linear = nn.Linear(ngf * 8, 1)
        self.stddev_group = 4
        self.stddev_feat = 1

    def forward(self, x, t, x_t):
        t_embed = self.act(self.t_embed(t))
        input_x = torch.cat((x, x_t), dim=1)
        h0 = self.start_conv(input_x)
        h1 = self.conv1(h0, t_embed)
        h2 = self.conv2(h1, t_embed)
        h3 = self.conv3(h2, t_embed)
        out = self.conv4(h3, t_embed)

        # Add statistical features
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], dim=1)

        # Final layers
        out = self.final_conv(out)
        out = self.act(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.end_linear(out)

        return out
    


def train_diff_gan(
        netG: UNet,
        netD: DiscriminatorTimeDependent,
        dataloader: DataLoader,
        optimizerG: torch.optim.Optimizer,
        optimizerD: torch.optim.Optimizer,
        timesteps: int = 4,
        epochs: int = 10,
        r1_gamma: float = 0.05,
        no_lr_decay: bool = False,
        device: torch.device = "cuda",
):
    
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, epochs, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, epochs, eta_min=1e-5)
    
    
    coeff = Diffusion_Coefficients(timesteps=timesteps, device=device)
    pos_coeff = Posterior_Coefficients(timesteps=timesteps, device=device)
    T = get_time_schedule(timesteps, device)
    
    global_step, epoch, init_epoch = 0, 0, 0
    
    
    for epoch in range(epochs):
       
        for iteration, (x, y) in enumerate(dataloader):
            for p in netD.parameters():  
                p.requires_grad = True  
        
            
            netD.zero_grad()
            
            #sample from p(x_0)
            real_data = x.to(device, non_blocking=True)
            
            #sample t
            t = torch.randint(0, timesteps, (real_data.size(0),), device=device)
            
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True
            
    
            # train with real
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)
            
            errD_real = F.softplus(-D_real)
            errD_real = errD_real.mean()
            
            errD_real.backward(retain_graph=True)
            
            
            grad_real = torch.autograd.grad(
                    outputs=D_real.sum(), inputs=x_t, create_graph=True
                    )[0]
            grad_penalty = (
                        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                        ).mean()
        
        
            grad_penalty = r1_gamma / 2 * grad_penalty
            grad_penalty.backward()

            # train with fake            
            x_0_predict = netG(x_tp1.detach(), t)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
                
            
            errD_fake = F.softplus(output)
            errD_fake = errD_fake.mean()
            errD_fake.backward()
    
            
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            
        
            #update G
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            
            
            t = torch.randint(0, timesteps, (real_data.size(0),), device=device)
            
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)            
           
            x_0_predict = netG(x_tp1.detach(), t)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
               
            
            errG = F.softplus(-output)
            errG = errG.mean()
            
            errG.backward()
            optimizerG.step()
                
           
            
            global_step += 1
            if iteration % 100 == 0:
                    print('epoch {} iteration{}, G Loss: {}, D Loss: {}'.format(epoch,iteration, errG.item(), errD.item()))
        
        if not no_lr_decay:
            
            schedulerG.step()
            schedulerD.step()
        
        if epoch % 10 == 0:
            print("Saving image")
            torchvision.utils.save_image(x_pos_sample, os.path.join("./training_saves/images", 'xpos_epoch_{}.png'.format(epoch)), normalize=True)
        
            x_t_1 = torch.randn_like(real_data)
            fake_sample = sample_from_model(pos_coeff, netG, timesteps, x_t_1)
            torchvision.utils.save_image(fake_sample, os.path.join("./training_saves/images", 'sample_discrete_epoch_{}.png'.format(epoch)), normalize=True)

            print('Saving content.')
            content = {'epoch': epoch + 1, 'global_step': global_step, 'args': {'timesteps': timesteps, 'r1_gamma': r1_gamma},
                        'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                        'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                        'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
            
            torch.save(content, os.path.join("./training_saves/weights", 'content_epoch_{}.pth'.format(epoch)))
            