import torch
import torch.nn.functional as F
import numpy as np


def extract(a, t, x_shape):
    """
    Extract values from a tensor at specified timesteps `t`.

    Args:
        a: Tensor of values, shape `[timesteps]`
        t: Tensor of timestep indices, shape `[batch_size]`
        x_shape: Shape of the input tensor `x` for reshaping

    Returns:
        Tensor of extracted values reshaped to `[batch_size, 1, 1, 1]`
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)  # Gather values at specific indices
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))  # Reshape to match input tensor shape


def q_sample(x_start, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    Add noise to the original image `x_start` at timestep `t`.

    Args:
        x_start: Original input image, shape `[batch_size, channels, height, width]`
        t: Timestep tensor, shape `[batch_size]`
        noise: Gaussian noise, shape `[batch_size, channels, height, width]`
        sqrt_alphas_cumprod: Precomputed cumulative product of alpha values
        sqrt_one_minus_alphas_cumprod: Precomputed sqrt(1 - alpha_cumprod)

    Returns:
        Noised input `x_t` at timestep `t`
    """
    # Extract coefficients for the current timestep
    sqrt_alpha = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alpha = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    # Return the noised sample
    return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise


@torch.no_grad()
def p_sample(model, x, t, t_index, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas_cumprod):
    """
    Sample from the reverse process of DDPM at a given timestep.

    Args:
        model: The U-Net denoising model
        x: Noised image at timestep `t`, shape `[batch_size, channels, height, width]`
        t: Current timestep tensor, shape `[batch_size]`
        t_index: Index of the timestep
        betas: Beta schedule for diffusion
        sqrt_one_minus_alphas_cumprod: Precomputed sqrt(1 - alpha_cumprod)
        sqrt_recip_alphas_cumprod: Precomputed reciprocal sqrt(alpha_cumprod)

    Returns:
        The denoised image at the previous timestep `t-1`
    """
    # Predict the noise using the model
    noise_pred = model(x, t)

    # Extract necessary coefficients
    beta = extract(betas, t, x.shape)
    sqrt_one_minus_alpha = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alpha = extract(sqrt_recip_alphas_cumprod, t, x.shape)

    # Predict the denoised `x0`
    x0_pred = sqrt_recip_alpha * (x - beta * noise_pred / sqrt_one_minus_alpha)

    # Compute posterior mean
    mean = x0_pred
    if t_index > 0:
        # Sample noise for timesteps > 1
        noise = torch.randn_like(x)
        return mean + torch.sqrt(beta) * noise
    else:
        # Return the final denoised result
        return mean


@torch.no_grad()
def p_sample_loop(model, shape, timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas_cumprod, device):
    """
    Perform the reverse process (denoising) to generate a sample image.

    Args:
        model: The U-Net denoising model
        shape: Shape of the output image `[batch_size, channels, height, width]`
        timesteps: Total number of diffusion steps
        betas: Beta schedule for diffusion
        sqrt_one_minus_alphas_cumprod: Precomputed sqrt(1 - alpha_cumprod)
        sqrt_recip_alphas_cumprod: Precomputed reciprocal sqrt(alpha_cumprod)
        device: PyTorch device (e.g., 'cpu' or 'cuda')

    Returns:
        Generated denoised image
    """
    # Start with pure Gaussian noise
    img = torch.randn(shape, device=device)
    for i in reversed(range(timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas_cumprod)
    return img


def get_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Generate a linear beta schedule for diffusion.

    Args:
        timesteps: Total number of diffusion steps
        beta_start: Starting value of beta
        beta_end: Ending value of beta

    Returns:
        Beta schedule tensor
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def precompute_alphas(betas):
    """
    Precompute alpha values and their cumulative products for diffusion.

    Args:
        betas: Beta schedule for diffusion

    Returns:
        A dictionary containing:
        - alphas_cumprod: Cumulative product of alpha values
        - sqrt_alphas_cumprod: Square root of alpha_cumprod
        - sqrt_one_minus_alphas_cumprod: Square root of (1 - alpha_cumprod)
        - sqrt_recip_alphas_cumprod: Reciprocal sqrt(alpha_cumprod)
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod),
    }

