import torch.nn as nn
from ifid.vae.utils import instantiate_from_config as ifid_instantiate_from_config
import torch

class VAECommon(nn.Module):
    def __init__(self, vae_config, resolution, channel_dim, downsample_factor):
        super().__init__()
        self.resolution = resolution
        self.channel_dim = channel_dim
        self.downsample_factor = downsample_factor
        self.vae = ifid_instantiate_from_config(vae_config)

    @property
    def latent_dim(self) -> int:
        """Return latent channels for compatibility with RAE interface."""
        return self.channel_dim

    @property
    def patch_size(self) -> int:
        """Return effective patch size (downsample factor) for compatibility."""
        return self.downsample_factor

    @property
    def hidden_size(self) -> int:
        """Alias for latent_dim for compatibility."""
        return self.channel_dim

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input images for VAE.

        Args:
            x: Images in [0, 1] range, shape (B, 3, H, W)

        Returns:
            Images in [-1, 1] range, resized to target resolution
        """
        # Resize if needed
        _, _, h, w = x.shape
        if h != self.resolution or w != self.resolution:
            x = nn.functional.interpolate(
                x,
                size=(self.resolution, self.resolution),
                mode='bilinear',
                align_corners=False
            )

        # Convert from [0, 1] to [-1, 1]
        x = x * 2.0 - 1.0 
        return x

    def forward(self, x, return_latent=False, return_posterior=False, enable_grad=False):
        x = self._preprocess(x)
        with torch.set_grad_enabled(enable_grad):
            z, vae_log = self.vae(x)
            posterior, xhat = vae_log["posterior"], vae_log["xhat"]
        xhat = ((xhat + 1.0) / 2.0)
        if return_latent and return_posterior:
            return xhat, z, posterior
        elif return_latent:
            return xhat, z
        elif return_posterior:
            return xhat, posterior
        else:
            return xhat

    def encode(self, x, enable_grad=False):
        x = self._preprocess(x)
        with torch.set_grad_enabled(enable_grad):
            return self.vae.encode(x)

    def decode(self, z, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            xhat = self.vae.decode(z)
            xhat = ((xhat + 1.0) / 2.0)
            return xhat