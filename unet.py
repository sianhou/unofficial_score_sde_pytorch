import torch
import torch.nn as nn

from layers import ResBlock, Downsample, get_timestep_embedding, Swish, Upsample


class Unet(nn.Module):
    def __init__(self, in_channels, nf=128, ch_mult=(1, 2, 2, 2), num_res_blocks=2,
                 attn_resolutions=(False, True, False, False), dropout=0.1, num_groups=32, resamp_with_conv=True):
        super(Unet, self).__init__()

        self.nf = nf
        self.tdim = self.nf * 4
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv
        self.num_groups = num_groups

        # TimeEmbedding
        self.temb_blocks = nn.Sequential(
            nn.Linear(self.nf, self.tdim),
            Swish(),
            nn.Linear(self.tdim, self.tdim),
        )

        # Head
        self.head = nn.Conv2d(in_channels=in_channels, out_channels=self.nf, kernel_size=3, stride=1, padding=1)

        # Downsampling block
        self.down_blocks = nn.ModuleList()
        out_ch = now_ch = self.nf
        chs = [out_ch]

        for i, mult in enumerate(self.ch_mult):
            out_ch = self.nf * mult
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                self.down_blocks.append(
                    ResBlock(now_ch, out_ch, self.tdim, num_groups=self.num_groups, dropout=self.dropout,
                             attn=self.attn_resolutions[i]))
                now_ch = out_ch
                chs.append(now_ch)

            if i != len(self.ch_mult) - 1:
                self.down_blocks.append(Downsample(now_ch, self.resamp_with_conv))
                chs.append(now_ch)

        # Middle
        self.middle_blocks = nn.ModuleList()
        self.middle_blocks.append(
            ResBlock(now_ch, now_ch, self.tdim, num_groups=self.num_groups, dropout=self.dropout, attn=True))
        self.middle_blocks.append(
            ResBlock(now_ch, now_ch, self.tdim, num_groups=self.num_groups, dropout=self.dropout, attn=False))

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(self.ch_mult))):
            out_ch = self.nf * mult
            for i_block in range(self.num_res_blocks + 1):
                self.up_blocks.append(
                    ResBlock(chs.pop() + now_ch, out_ch, self.tdim, num_groups=self.num_groups, dropout=self.dropout,
                             attn=self.attn_resolutions[i])
                )
                now_ch = out_ch
            if i != 0:
                self.up_blocks.append(Upsample(now_ch, self.resamp_with_conv))

        # Tail
        self.tail = nn.Sequential(
            nn.GroupNorm(self.num_groups, now_ch),
            Swish(),
            nn.Conv2d(in_channels=now_ch, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x, timesteps):
        # Timestep embedding
        temb = get_timestep_embedding(timesteps, self.nf)
        temb = self.temb_blocks(temb)

        # Head
        h = self.head(x)
        hs = [h]

        # Downsample blocks
        for block in self.down_blocks:
            h = block(h, temb)
            hs.append(h)

        # Middle blocks
        for block in self.middle_blocks:
            h = block(h, temb)

        # Upsample blocks
        for block in self.up_blocks:
            if isinstance(block, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, temb)

        # Tail
        h = self.tail(h)

        assert len(hs) == 0

        return h


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(128, 3, 32, 32).to(device)
    timesteps = torch.randint(0, 1000, (128,)).to(device)
    model = Unet(in_channels=3).to(device)
    out = model(x, timesteps)
    print(f"test ddpm: out.shape = {out.shape}")
