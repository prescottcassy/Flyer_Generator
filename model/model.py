import torch
import torch.nn as nn
import math

# This module defines the SDXL model architecture only. Dataset and environment
# initialization (seeding, dataloader creation) are deliberately NOT performed at
# import time to avoid side-effects. Callers should initialize datasets and seed
# explicitly (for example, from `train_model.py`).

# GELUConvBlock remains unchanged
class GELUConvBlock(nn.Module):
    """
    A convolutional block with GELU activation, GroupNorm, and two Conv2d layers.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel. Default is 3.
        padding (int): Padding added to all sides of the input. Default is 1.
        stride (int): Stride of the convolution. Default is 1.
    """    

    def __init__(self, in_channels, out_channels, group_size, kernel_size=3, padding=1, stride=1):
        super(GELUConvBlock, self).__init__()
        
        if out_channels % group_size != 0:
            group_size = min(group_size, out_channels)
            while out_channels % group_size != 0:
                group_size -= 1

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(num_groups=group_size, num_channels=out_channels)
        )

    def forward(self, x):
        # Assume the block modules are already on the correct device/dtype.
        # Moving modules inside forward is dynamic and prevents compilation.
        return self.block(x)

# Rearrange & Downsample Block
class RearrangePoolBlock(nn.Module):
    """
    A block that rearranges pixels to downsample the image and increase the channels by (downscale_factor^2).

    Args:
        downscale_factor (int): Factor by which to downsample the spatial dimensions. Default is 2.
        normalize (bool): Whether to apply normalization after rearranging. Default is True.
    """
    def __init__(self, in_channels, downscale_factor=2, normalize=True):
        super().__init__()
        self.downscale_factor = downscale_factor
        self.rearrange = nn.PixelUnshuffle(downscale_factor)
        self.normalize = normalize
        self.norm_layer = nn.BatchNorm2d(in_channels * (downscale_factor ** 2)) if normalize else nn.Identity()

    def forward(self, x):
        x = self.rearrange(x)
        x = self.norm_layer(x)
        return x

# Downsample block for SDXL architecture
class DownBlock(nn.Module):
    """
    A downsampling block that reduces spatial dimensions and increases channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        group_size (int): Number of groups for GroupNorm.
        downscale_factor (int): Factor by which to downsample the spatial dimensions.
    """
    def __init__(self, in_channels, out_channels, group_size, downscale_factor=2):
        super().__init__()
        self.conv = nn.Sequential(
            GELUConvBlock(in_channels, out_channels, group_size),
            GELUConvBlock(out_channels, out_channels, group_size)
        )
        self.pool = RearrangePoolBlock(in_channels=out_channels, downscale_factor=downscale_factor)
        self.channel_reduction_conv = nn.Conv2d(
            out_channels * (downscale_factor ** 2), 
            out_channels,
            kernel_size=1
        )

    # down block initialized

    def forward(self, x):
        skip = self.conv(x)
        
        x = self.pool(skip)
        
        pooled_x = self.channel_reduction_conv(x)
        return pooled_x, skip


# Rearrange Upsample block for SDXL architecture
class RearrangeUpsampleBlock(nn.Module):
    """
    An upsampling block that increases spatial dimensions and reduces channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        upscale_factor (int): Factor by which to upscale the spatial dimensions.
    """

    def __init__(self, in_channels, upscale_factor=2, normalize=True):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.rearrange = nn.PixelShuffle(upscale_factor)
        self.normalize = normalize
        self.norm_layer = nn.BatchNorm2d(in_channels // (upscale_factor ** 2)) if normalize else nn.Identity()

    def forward(self, x):
        x = self.rearrange(x)
        x = self.norm_layer(x)
        return x
# Upsampling block for our SDXL architecture
class Uplock(nn.Module):
    """
    An upsampling block that increases spatial dimensions and reduces channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        upscale_factor (int): Factor by which to upscale the spatial dimensions.
    """
    def __init__(self, in_channels, out_channels, skip_channels, group_size, upscale_factor=2):
        super().__init__()
        self.upsample = RearrangeUpsampleBlock(in_channels=in_channels, upscale_factor=upscale_factor)
        
        self.channel_increase_conv = nn.Conv2d(
            in_channels // (upscale_factor ** 2), 
            out_channels, 
            kernel_size=1
        )

        combined_channels = out_channels + skip_channels

        self.conv = nn.Sequential(
            GELUConvBlock(combined_channels, out_channels, group_size),
            GELUConvBlock(out_channels, out_channels, group_size)
        )
    # Uplock initialized

    def forward(self, x, skip):
        x = self.upsample(x)
        x = self.channel_increase_conv(x)
        x = torch.cat((x, skip), dim=1)  # Concatenate along channel dimension
        x = self.conv(x)
        return x
    

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal Positional Embedding.

    Args:
        dim (int): Dimension of the embedding.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # Validate input
        if time.dim() != 1:
            raise ValueError(f"Expected 1D tensor for 'time', but got shape {time.shape}")

        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)  # Use math.log for numerical stability
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class EmbedDrawingConditioning(nn.Module):
    """
    Embeds drawing conditioning information into a higher-dimensional space.

    Args:
        in_channels (int): Number of input channels.
        emb_dim (int): Dimension of the embedding space.
    """

    def __init__(self, in_channels, emb_dim):
        super().__init__()
        self.in_channels = in_channels
        self.embedding = nn.Embedding(in_channels, emb_dim)

        self.model = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.GELU(),
            nn.Unflatten(dim=1, unflattened_size=(emb_dim // 4, 2, 2))
        )
        self._init_weights()
    # EmbedDrawingConditioning initialized

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Validate input
        if x.dim() == 1:  # Class indices [B]
            x = x.long()  # Ensure it's Long type
        elif x.dim() == 2 and x.shape[1] == 1:  # Class indices [B, 1]
            x = x.squeeze(1).long()  # Ensure it's Long type after squeezing
        elif x.dim() == 2 and x.shape[1] > 1:  # Assume one-hot encoding
            if not torch.all((x == 0) | (x == 1)):
                raise ValueError("Input appears to be 2D but not one-hot encoded.")
            x = torch.argmax(x, dim=-1).long()  # Convert one-hot to indices
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected 1D indices or 2D one-hot encoding.")

        # Flatten input for embedding layer
        x = x.view(-1)  # Embedding layer expects [B]

        # Generate embeddings using the embedding layer first
        embedded_x = self.embedding(x)  # [B, emb_dim]
        return self.model(embedded_x)
    
class SDXL(nn.Module):
    def __init__(self, img_channels, img_size, down_channels, t_embed_dim, c_embed_dim, num_classes: int = 0, device=None):
        super().__init__()

        # Save constructor arguments as attributes
        self.img_channels = img_channels
        self.img_size = img_size
        self.down_channels = down_channels
        self.t_embed_dim = t_embed_dim
        self.c_embed_dim = c_embed_dim
        self.num_classes = num_classes

        # Store model channel dimension for later use
        self.model_channels = down_channels[-1]

        # Time and Class Embeddings
        self.time_embedding = nn.Sequential(
            nn.Linear(t_embed_dim, self.model_channels),
            nn.LayerNorm(self.model_channels)
        )
        self.class_embedding = nn.Sequential(
            nn.Linear(c_embed_dim, self.model_channels),
            nn.LayerNorm(self.model_channels)
        )

        # Map class indices to a c_embed_dim vector
        num_classes = max(1, int(num_classes or 0))
        self.class_mlp = nn.Embedding(num_classes, c_embed_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_embed_dim),
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.GELU()
        )

        # Initial Convolution
        self.init_conv = GELUConvBlock(img_channels, down_channels[0], group_size=8)

        # Downsampling Blocks
        self.downs = nn.ModuleList()
        for i in range(len(down_channels) - 1):
            self.downs.append(
                DownBlock(
                    in_channels=down_channels[i],
                    out_channels=down_channels[i + 1],
                    group_size=8,
                    downscale_factor=2
                )
            )

        # Middle Block
        self.middle = nn.Sequential(
            GELUConvBlock(down_channels[-1], down_channels[-1], group_size=8),
            nn.Dropout(0.1),
            GELUConvBlock(down_channels[-1], down_channels[-1], group_size=8)
        )

        # Upsampling Blocks
        self.ups = nn.ModuleList()
        num_downs = len(down_channels) - 1
        for i in range(num_downs - 1, -1, -1):
            in_channels = down_channels[i + 1]
            out_channels = down_channels[i]
            skip_channels = down_channels[i + 1]
            self.ups.append(
                Uplock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    skip_channels=skip_channels,
                    group_size=8,
                    upscale_factor=2
                )
            )

        # Final Convolution
        self.final_conv = nn.Conv2d(down_channels[0], img_channels, kernel_size=1)

        # Move model to the specified device (done once at init)
        if device is not None:
            self.to(device)


    def forward(self, x, t, c):
        """
        Forward pass for the SDXL model.

        Args:
            x (torch.Tensor): Input image tensor.
            t (torch.Tensor): Time step tensor.
            c (torch.Tensor): Class embedding tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Replace state_dict() with a direct loop over self.parameters()
        param_list = []
        for param in self.parameters():
            if param is not None:
                param_list.append(param)

        # Ensure `c` is initialized properly
        if c is None:
            c = torch.zeros((x.size(0), 1), device=param_list[0].device, dtype=torch.long)
        else:
            # Move to device first, then cast to long. Avoid passing dtype kwargs to .to()
            c = c.to(param_list[0].device)
            if c.dtype != torch.long:
                c = c.long()

    # Input tensors are expected to be moved to `device` by the caller

        # Ensure all tensors are moved to the same device at the beginning of the forward method
        device = self.time_embedding[0].weight.device if isinstance(self.time_embedding[0].weight.device, torch.device) else torch.device('cpu')
        x = x.to(device)
        t = t.to(device)
        c = c.to(device)

        # Time and class embeddings
        t_emb = self.time_mlp(t)
        c_emb = self.class_mlp(c)

        # Initial convolution
        x = self.init_conv(x)

        # Downsampling path
        skips = []
        for down in self.downs:
            x, skip = down(x)
            skips.append(skip)

        # Middle block
        x = self.middle(x)

        # Upsampling path
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        # Final convolution
        x = self.final_conv(x)

        return x
