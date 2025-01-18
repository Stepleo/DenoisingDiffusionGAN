import argparse
import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from datasets_prep.lsun import LSUN
from datasets_prep.stackmnist_data import StackedMNIST, _data_transforms_stacked_mnist
from datasets_prep.stack_mnist2D import StackedMNIST2D, _data_transforms_stacked_mnist_2d
from datasets_prep.lmdb_datasets import LMDBDataset
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))
    
#%% Diffusion coefficients 
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

class Diffusion_Coefficients():
    def __init__(self, args, device):
                
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)
    
def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)
      
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    
    return x_t

def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    
    return x_t, x_t_plus_one
#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
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

def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
        
    return x

#%%
def train(gpu, args):
    from score_sde.models.discriminator import Discriminator_small, Discriminator_large
    from score_sde.models.ncsnpp_generator_adagn import NCSNpp
    from EMA import EMA
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda:{}'.format(gpu))
    
    batch_size = args.batch_size
    nz = args.nz  # latent dimension
    
    if args.dataset == 'cifar10':
        dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)
    elif args.dataset == 'stackmnist':
        train_transform, valid_transform = _data_transforms_stacked_mnist()
        dataset = StackedMNIST(root='./data', train=True, download=False, transform=train_transform, subset_ratio=0.2)
    elif args.dataset == 'stackmnist2D':
        train_transform, valid_transform = _data_transforms_stacked_mnist_2d()
        dataset = StackedMNIST2D(root='./data', train=True, download=True, transform=train_transform, subset_ratio=0.2)
    elif args.dataset == 'lsun':
        train_transform = transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.CenterCrop(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
        train_data = LSUN(root='/datasets/LSUN/', classes=['church_outdoor_train'], transform=train_transform)
        subset = list(range(0, 120000))
        dataset = torch.utils.data.Subset(train_data, subset)
    elif args.dataset == 'celeba_256':
        train_transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        dataset = LMDBDataset(root='/datasets/celeba-lmdb/', name='celeba', train=True, transform=train_transform)
    
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True,
                                               drop_last=True)
    # print(f"Number of iterations per epoch: {len(data_loader)}")
    
    netG = NCSNpp(args).to(device)

    if args.dataset == 'cifar10' or args.dataset == 'stackmnist':    
        netD = Discriminator_small(nc=2*args.num_channels, ngf=args.ngf,
                               t_emb_dim=args.t_emb_dim,
                               act=nn.LeakyReLU(0.2)).to(device)
    else:
        netD = Discriminator_large(nc=2*args.num_channels, ngf=args.ngf, 
                                   t_emb_dim=args.t_emb_dim,
                                   act=nn.LeakyReLU(0.2)).to(device)
    
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    
    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)
    
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch, eta_min=1e-5)

    exp = args.exp
    parent_dir = "./saved_info/dd_gan/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        copy_source(__file__, exp_path)
        shutil.copytree('score_sde/models', os.path.join(exp_path, 'score_sde/models'))

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)
    
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        netG.load_state_dict(checkpoint['netG_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0
    
    for epoch in tqdm(range(init_epoch, args.num_epoch + 1), desc="Epochs"):
    # Wrap the inner loop with tqdm, adding `leave=False` to not leave the progress bar after completion
        for iteration, (x, y) in tqdm(enumerate(data_loader), desc="Iterations", leave=False):
            for p in netD.parameters():  
                p.requires_grad = True
            netD.zero_grad()
            
            real_data = x.to(device, non_blocking=True)
            # print(f"real_data shape: {real_data.shape}")
            log_plot(epoch, iteration, real_data[0],'unknown', 'real_data', 0)
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            # print(f"t shape: {t.shape}")
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True

            # print(f"x_t shape: {x_t.shape}")
            log_plot(epoch, iteration, x_t[0],t[0], 'x_t', 0)
            # print(f"x_tp1 shape: {x_tp1.shape}")
            log_plot(epoch, iteration, x_tp1[0],t[0], 'x_tp1', 0)
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)
            # print(f"D_real shape: {D_real.shape}")
            errD_real = F.softplus(-D_real)
            log_info(epoch, iteration, errD_real[0],'D_real')
            errD_real = errD_real.mean()
            errD_real.backward(retain_graph=True)
            
            grad_real = torch.autograd.grad(outputs=D_real.sum(), inputs=x_t, create_graph=True)[0]
            # print(f"grad_real shape: {grad_real.shape}")
            # plot_image_with_channels(grad_real[0], "grad_real")
            grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
            # print(f"grad_penalty shape: {grad_penalty.shape}")
            
            grad_penalty = args.r1_gamma / 2 * grad_penalty
            grad_penalty.backward()

            latent_z = torch.randn(batch_size, nz, device=device)
            # print(f"latent_z shape: {latent_z.shape}")
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            # print(f"x_0_predict shape: {x_0_predict.shape}")
            log_plot(epoch, iteration, x_0_predict[0],t[0], 'x_0_predict', 0)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            # print(f"x_pos_sample shape: {x_pos_sample.shape}")
            log_plot(epoch, iteration, x_pos_sample[0],t[0], 'x_pos_sample', 0)
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            # print(f"output shape: {output.shape}")
            errD_fake = F.softplus(output)
            log_info(epoch, iteration, errD_fake[0],'D_fake')
            errD_fake = errD_fake.mean()
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizerD.step()
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()

            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)

            # print(f"x_t shape: {x_t.shape}")
            log_plot(epoch, iteration, x_t[0],t[0], 'x_t', 1)
            # print(f"x_tp1 shape: {x_tp1.shape}")
            log_plot(epoch, iteration, x_tp1[0],t[0], 'x_tp1', 1)
            latent_z = torch.randn(batch_size, nz, device=device)
            # print(f"latent_z shape: {latent_z.shape}")
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            # print(f"x_0_predict shape: {x_0_predict.shape}")
            log_plot(epoch, iteration, x_0_predict[0],t[0], 'x_0_predict', 1)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            # print(f"x_pos_sample shape: {x_pos_sample.shape}")
            log_plot(epoch, iteration, x_pos_sample[0],t[0], 'x_pos_sample', 1)
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            # print(f"output shape: {output.shape}")  
            errG = F.softplus(-output)
            log_info(epoch, iteration, errG[0],'G')
            errG = errG.mean()
            errG.backward()
            optimizerG.step()

            
            global_step += 1
            if iteration % 100 == 0:
                print(f'epoch {epoch} iteration {iteration}, G Loss: {errG.item()}, D Loss: {errD.item()}')
        
        if not args.no_lr_decay:
            schedulerG.step()
            schedulerD.step()

        if epoch % 1 == 0:
            torchvision.utils.save_image(x_pos_sample, os.path.join(exp_path, f'xpos_epoch_{epoch}.png'), normalize=True)
            x_t_1 = torch.randn_like(real_data)
            fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, args)
            torchvision.utils.save_image(fake_sample, os.path.join(exp_path, f'sample_discrete_epoch_{epoch}.png'), normalize=True)

        if args.save_content and epoch % args.save_content_every == 0:
            print('Saving content.')
            content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                       'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                       'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                       'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
            torch.save(content, os.path.join(exp_path, 'content.pth'))

        if epoch % args.save_ckpt_every == 0:
            if args.use_ema:
                optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
            torch.save(netG.state_dict(), os.path.join(exp_path, f'netG_{epoch}.pth'))
            if args.use_ema:
                optimizerG.swap_parameters_with_ema(store_params_in_ema=True)


def init_processes(gpu, args):
    """ Initialize the environment and set up the training loop. """
    torch.cuda.set_device(gpu)
    train(gpu, args)

def cleanup():
    """ Cleanup function (No DDP to clean up). """
    pass

def log_info(epoch, iteration, loss,type):
    """ Log the training information. """
    if iteration % 100 == 0:
        filename=f'./saved_info/dd_gan/{args.dataset}/{args.exp}/plots/epoch_{epoch}_iteration_{iteration}_loss.txt'
        with open(filename, 'a') as f:
            if type == 'D_real':
                f.write(f'epoch {epoch} iteration {iteration}, D_real Loss: {loss.item()}')
            if type == 'D_fake':
                f.write(f'epoch {epoch} iteration {iteration}, D_fake Loss: {loss.item()}')
            if type == 'G':
                f.write(f'epoch {epoch} iteration {iteration}, G Loss: {loss.item()}')
    
        
        
    return

def log_plot(epoch, iteration, image,timestep, graph_to_plot, rank_of_train_loop):
    """ Log the plot of the graph. """
    if iteration % 100 == 0:
        if graph_to_plot == 'real_data':
            title = 'Real Data'
            step=0
        elif graph_to_plot == 'x_t':
            title = f'X_t with t={timestep}'
            step=1+4*rank_of_train_loop
        elif graph_to_plot == 'x_tp1':
            title = f'X_t+1 with t={timestep} (This is {timestep+1}th timestep)'
            step=2+4*rank_of_train_loop
        elif graph_to_plot == 'x_0_predict':
            title = f"Gan prediction x_0' from t={timestep}"
            step=3+4*rank_of_train_loop
        elif graph_to_plot == 'x_pos_sample':
            title = f"Sampled X_t' from X_0', t={timestep}"
            step=4+4*rank_of_train_loop
        else:
            raise ValueError(f'Unknown graph to plot: {graph_to_plot}')
        
        #normalize the image
        image = (image - image.min()) / (image.max() - image.min())
        plot=plot_image_with_channels(image, title)
        plt.close()    
        #save the plot
        name = f'epoch_{epoch}_iteration_{iteration}_step_{step}_image_{graph_to_plot}_time_{timestep}.png'

    
        
        os.makedirs(f'./saved_info/dd_gan/{args.dataset}/{args.exp}/plots', exist_ok=True)
        plot.savefig(f'./saved_info/dd_gan/{args.dataset}/{args.exp}/plots/{name}')
    return
    


def plot_image_with_channels(image,title):
    """
    Plot an RGB image and its individual R, G, B channels.
    
    Args:
    - image (torch.Tensor): A 3-channel RGB image of shape (3, height, width).
    """
    # Ensure the image has the correct shape
    if image.shape[0] != 3:
        raise ValueError("The input image must have 3 channels (RGB).")
    
    # Detach the tensor and convert it to a NumPy array
    image_np = image.detach().cpu().numpy()
    
    # Extract the channels
    r, g, b = image_np[0], image_np[1], image_np[2]
    
    # Create the figure and subplots
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    
    # Plot the RGB image
    rgb_image = np.transpose(image_np, (1, 2, 0))  # (3, height, width) -> (height, width, 3)
    axes[0].imshow(rgb_image)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # Plot the Red channel
    axes[1].imshow(r, cmap='Reds')
    axes[1].set_title('Red Channel')
    axes[1].axis('off')
    
    # Plot the Green channel
    axes[2].imshow(g, cmap='Greens')
    axes[2].set_title('Green Channel')
    axes[2].axis('off')
    
    # Plot the Blue channel
    axes[3].imshow(b, cmap='Blues')
    axes[3].set_title('Blue Channel')
    axes[3].axis('off')
    
    # Set the title of the figure
    fig.suptitle(title)
    
    
    # Display the plots
    plt.tight_layout()
    # plt.show()
    return fig

if __name__ == '__main__':
        
    # Argument parser setup
    parser = argparse.ArgumentParser('ddgan parameters')

    # General and Experiment Parameters
    parser.add_argument('--seed', type=int, default=1024, help='Seed used for initialization')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--image_size', type=int, default=32, help='Size of image')
    parser.add_argument('--num_channels', type=int, default=3, help='Channel of image')
    parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1, help='Beta min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20., help='Beta max for diffusion')

    # Model-specific Parameters
    parser.add_argument('--num_channels_dae', type=int, default=128, help='Initial channels in denoising model')
    parser.add_argument('--n_mlp', type=int, default=3, help='Number of MLP layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int, help='Channel multipliers')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of ResNet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), help='Resolutions of applying attention')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True, help='Use conv for resampling')
    parser.add_argument('--conditional', action='store_false', default=True, help='Noise conditional')
    parser.add_argument('--fir', action='store_false', default=True, help='Use FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True, help='Skip rescaling')
    parser.add_argument('--resblock_type', default='biggan', help='ResNet block type')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'], help='Progressive output type')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'], help='Progressive input type')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'], help='Progressive combine method')

    # Time and embedding Parameters
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'], help='Time embedding type')
    parser.add_argument('--fourier_scale', type=float, default=16., help='Fourier transform scale')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # Generator and Training Parameters
    parser.add_argument('--exp', default='experiment_cifar_default', help='Experiment name')
    parser.add_argument('--dataset', default='cifar10', help='Dataset name')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='Learning rate for generator')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='Learning rate for discriminator')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='Beta2 for Adam optimizer')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)
    parser.add_argument('--use_ema', action='store_true', default=False, help='Use EMA (Exponential Moving Average)')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='Decay rate for EMA')
    parser.add_argument('--r1_gamma', type=float, default=0.05, help='Coefficient for R1 regularization')
    parser.add_argument('--lazy_reg', type=int, default=None, help='Lazy regularization')

    # Saving Parameters
    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=50, help='Save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=25, help='Save checkpoint every x epochs')

    # No need for Distributed Data Parallel (DDP) arguments anymore
    args = parser.parse_args()

    # GPU setup (just a single GPU setup, no multi-process DDP)
    gpu = 0  # Use GPU 0 by default
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # Start training
    init_processes(gpu, args)