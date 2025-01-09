import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# Loss function for denoising
def p_losses(denoise_model, x_start, t, cond, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, cond)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


# Position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# Denoise model
class DenoiseNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, cond_input_dim, d_cond):
        super(DenoiseNN, self).__init__()
        self.n_layers = n_layers
        self.cond_input_dim = cond_input_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_input_dim, d_cond),
            nn.ReLU(),
            nn.Linear(d_cond, d_cond),
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        mlp_layers = [nn.Linear(input_dim+d_cond, hidden_dim)] + [nn.Linear(hidden_dim+d_cond, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, input_dim))
        self.mlp = nn.ModuleList(mlp_layers)

        bn_layers = [nn.BatchNorm1d(hidden_dim) for i in range(n_layers-1)]
        self.bn = nn.ModuleList(bn_layers)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, t, cond):
        cond = torch.reshape(cond, (-1, self.cond_input_dim))
        cond = torch.nan_to_num(cond, nan=-100.0)
        cond = self.cond_mlp(cond)
        t = self.time_mlp(t)
        for i in range(self.n_layers-1):
            x = torch.cat((x, cond), dim=1)
            x = self.relu(self.mlp[i](x))+t
            x = self.bn[i](x)
        x = self.mlp[self.n_layers-1](x)
        return x


class ImprovedDenoiseNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, cond_input_dim, d_cond):
        super(ImprovedDenoiseNN, self).__init__()
        self.n_layers = n_layers
        self.cond_input_dim = cond_input_dim

        # Enhanced conditioning network with residual connections
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_input_dim, d_cond),
            nn.LayerNorm(d_cond),
            nn.ReLU(),
            nn.Linear(d_cond, d_cond),
            nn.LayerNorm(d_cond),
            nn.ReLU(),
            nn.Linear(d_cond, d_cond),
        )

        # FiLM conditioning layers
        self.film_gamma = nn.ModuleList([
            nn.Linear(d_cond, hidden_dim) for _ in range(n_layers - 1)
        ])
        self.film_beta = nn.ModuleList([
            nn.Linear(d_cond, hidden_dim) for _ in range(n_layers - 1)
        ])

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-attention layers for better conditioning integration
        self.cross_attention = nn.ModuleList([
            CrossAttention(hidden_dim, d_cond) for _ in range(n_layers - 1)
        ])

        # Main network with residual connections
        mlp_layers = [nn.Linear(input_dim, hidden_dim)]
        for i in range(n_layers - 2):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
        mlp_layers.append(nn.Linear(hidden_dim, input_dim))
        self.mlp = nn.ModuleList(mlp_layers)

        self.layer_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers - 1)
        ])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, t, cond):
        # Preprocess conditioning
        cond = torch.reshape(cond, (-1, self.cond_input_dim))
        cond = torch.nan_to_num(cond, nan=-100.0)
        cond = self.cond_mlp(cond)

        # Time embedding
        t = self.time_mlp(t)

        # Main forward pass with enhanced conditioning
        h = x
        for i in range(self.n_layers - 1):
            # Residual connection
            residual = h

            # Apply main layer
            h = self.mlp[i](h)

            # Apply FiLM conditioning
            gamma = self.film_gamma[i](cond)
            beta = self.film_beta[i](cond)
            h = h * gamma.unsqueeze(1) + beta.unsqueeze(1)

            # Add time embedding
            h = h + t

            # Cross-attention with conditioning
            h = self.cross_attention[i](h, cond)

            # Layer norm and activation
            h = self.layer_norm[i](h)
            h = self.relu(h)
            h = self.dropout(h)

            # Add residual connection
            if h.shape == residual.shape:
                h = h + residual

        # Final layer
        x = self.mlp[self.n_layers - 1](h)
        return x


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, d_cond):
        super().__init__()
        self.scale = hidden_dim ** -0.5
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(d_cond, hidden_dim)
        self.to_v = nn.Linear(d_cond, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, cond):
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        return self.to_out(out)


@torch.no_grad()
def p_sample(model, x, t, cond, t_index, betas):
    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, cond) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, cond, timesteps, betas, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, timesteps)):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), cond, i, betas)
        imgs.append(img)
        #imgs.append(img.cpu().numpy())
    return imgs



@torch.no_grad()
def sample(model, cond, latent_dim, timesteps, betas, batch_size):
    return p_sample_loop(model, cond, timesteps, betas, shape=(batch_size, latent_dim))
