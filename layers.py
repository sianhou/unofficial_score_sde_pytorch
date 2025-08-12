import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnBlockEfficient(nn.Module):
    def __init__(self, channels, num_groups=32, dropout=0.0):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.q = Nin(channels, channels)
        self.k = Nin(channels, channels)
        self.v = Nin(channels, channels)
        self.proj = Nin(channels, channels, init_scale=0.0)
        self.dropout = dropout

        self.init_weights()

    def init_weights(self):
        for m in [self.q, self.k, self.v, self.proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x, temb=None):
        """
        Memory-efficient attention block.
        Input:  x [B, C, H, W]
        Output: same shape as input
        """
        B, C, H, W = x.shape
        h = self.norm(x)

        # Project Q, K, V and reshape to [B, HW, C]
        q = self.q(h).flatten(2).transpose(1, 2)  # [B, HW, C]
        k = self.k(h).flatten(2).transpose(1, 2)  # [B, HW, C]
        v = self.v(h).flatten(2).transpose(1, 2)  # [B, HW, C]

        # Use PyTorch's optimized scaled dot product attention
        # This avoids materializing the full [B, HW, HW] matrix
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout,
            is_causal=False
        )  # [B, HW, C]

        # Restore shape back to [B, C, H, W]
        h = attn_out.transpose(1, 2).view(B, C, H, W)

        # Output projection and residual connection
        h = self.proj(h)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.q = Nin(channels, channels)
        self.k = Nin(channels, channels)
        self.v = Nin(channels, channels)
        self.proj = Nin(channels, channels, init_scale=0.0)

        self.init_weights()

    def init_weights(self):
        for m in [self.q, self.k, self.v, self.proj]:
            nn.init.xavier_uniform_(m.nin.weight)
            nn.init.zeros_(m.nin.bias)
        nn.init.xavier_uniform_(self.proj.nin.weight, gain=1e-5)

    def forward(self, x, temb=None):
        B, C, H, W = x.shape  # PyTorch uses [B, C, H, W]
        h = self.norm(x)

        q = self.q(h)  # [B, C, H, W]
        k = self.k(h)  # [B, C, H, W]
        v = self.v(h)  # [B, C, H, W]

        # Compute attention weights
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)  # [B, HW, C]
        k = k.view(B, C, H * W)  # [B, C, HW]
        w = torch.bmm(q, k) * (C ** -0.5)  # [B, HW, HW]
        w = F.softmax(w, dim=-1)

        # Apply attention
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)  # [B, HW, C]
        h = torch.bmm(w, v)  # [B, HW, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        h = self.proj(h)

        return x + h


class Nin(nn.Module):
    def __init__(self, in_channels, out_channels, init_scale=0.1):
        super().__init__()
        self.nin = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.nin.weight)
        nn.init.zeros_(self.nin.bias)

    def forward(self, x):
        return self.nin(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, tdim, num_groups=32, dropout=0.1, attn=True):
        super().__init__()

        # GroupNorm: Normalizes channels in groups → stabilizes training, especially for small batch sizes.
        # Swish: Smooth activation function(x * sigmoid(x)), often better than ReLU.
        # Conv2d(3×3): Extracts spatial features.
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            Swish(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )

        # Maps time embedding vector temb(shape[B, tdim]) to channel dimension[B, out_ch].
        # This is broadcasted over spatial dimensions and added to h for conditional modulation.
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_channels),
        )

        # GroupNorm + Swish → normalized & activated features
        # Dropout → regularization
        # Conv2d(3×3): Extracts spatial features.
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_channels),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )

        if in_channels != out_channels:
            self.shortcut = Nin(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

        if attn:
            self.attn = AttnBlock(out_channels)
        else:
            self.attn = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb=None):
        h = self.block1(x)

        if temb is not None:
            h += self.temb_proj(temb)[:, :, None, None]

        h = self.block2(h)

        h = h + self.shortcut(x)

        h = self.attn(h)

        return h


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimestepEmbedding(nn.Module):
    def __init__(self, num_timesteps, embedding_dim, max_positions=10000):
        super().__init__()
        emb = torch.arange(0, embedding_dim, 2) / embedding_dim * math.log(max_positions)
        emb = torch.exp(-emb)
        pos = torch.arange(num_timesteps)
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(num_timesteps, embedding_dim)

        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
        )

    def forward(self, t):
        return self.time_embedding(t)


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(timesteps.device)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
    emb = emb.view(timesteps.shape[0], embedding_dim)
    return emb


class Downsample(nn.Module):
    def __init__(self, in_ch, with_conv=True):
        super().__init__()

        self.with_conv = with_conv

        if with_conv:
            self.down = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        else:
            self.down = nn.AvgPool2d(2)

        self.init_weights()

    def init_weights(self):
        if self.with_conv:
            nn.init.xavier_uniform_(self.down.weight)
            nn.init.zeros_(self.down.bias)

    def forward(self, x, temb=None):
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, with_conv=True):
        super().__init__()

        self.with_conv = with_conv

        if with_conv:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
            )

        self.init_weights()

    def init_weights(self):
        if self.with_conv:
            nn.init.xavier_uniform_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x, temb=None):
        return self.up(x)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # test attention block
    attn = AttnBlock(channels=64, normalize=nn.GroupNorm).to(device)
    x = torch.randn(8, 64, 32, 32).to(device)
    out = attn(x)
    print(out.shape)

    # test resnet block
    tdim = 512
    temb = torch.randint(0, 1000, (8, tdim)).to(device)
    print(f"test resnet block: temb.shape = {temb.shape}")
    res = ResBlock(in_channels=64, out_channels=64, tdim=tdim).to(device)
    x = torch.randn(8, 64, 32, 32).to(device)
    out = res(x, temb=temb)
    print(f"test resnet block: out.shape = {out.shape}")

    # test time embedding
    x = torch.arange(0, 1000)
    model = TimestepEmbedding(num_timesteps=1000, embedding_dim=64, max_positions=10000)
    out1 = model(x)
    out2 = get_timestep_embedding(x, 64)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))  # 2行1列，整体高点

    im1 = axes[0].imshow(out1.numpy(), aspect='auto', cmap='viridis', origin='lower')
    axes[0].set_title('Positional Encoding (Time Embedding) - Model')
    axes[0].set_xlabel('Embedding Dimension')
    axes[0].set_ylabel('Time Step')
    fig.colorbar(im1, ax=axes[0], orientation='vertical', label='Embedding Value')

    im2 = axes[1].imshow(out2.numpy(), aspect='auto', cmap='viridis', origin='lower')
    axes[1].set_title('Positional Encoding (Time Embedding) - Function')
    axes[1].set_xlabel('Embedding Dimension')
    axes[1].set_ylabel('Time Step')
    fig.colorbar(im2, ax=axes[1], orientation='vertical', label='Embedding Value')

    plt.tight_layout()
    plt.show()
