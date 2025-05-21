import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from torch import nn
from torch.nn import functional as F
import numbers
from einops import rearrange

from modules.normalization import PCN


class ResBlock(nn.Module):
    def __init__(self, channels_out, kernel_size, channels_in=None, stride=1, dilation=1, padding=1, use_dropout=False):
        super(ResBlock, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == channels_out:
            channels_in = channels_out
            self.projection = None
        else:
            self.projection = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, dilation=1)
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.elu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.n2 = nn.BatchNorm2d(channels_out)
        if self.use_dropout:
            self.dropout = nn.Dropout()
        self.elu2 = nn.ELU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.elu1(out)
        out = self.conv2(out)
        # out = self.n2(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out = out + residual
        out = self.elu2(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride, padding=0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.n1 = nn.BatchNorm2d(channels_out)
        self.elu1 = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.n1(out)
        out = self.elu1(out)
        return out


class PCBlock(nn.Module):
    def __init__(self, channels_out, kernel_size, channels_in=None, stride=1, dilation=1, padding=1, use_dropout=False):
        super(PCBlock, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == channels_out:
            channels_in = channels_out
            self.projection = None
        else:
            self.projection = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, dilation=1)
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.elu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.pcn = PCN(channels_out)
        if self.use_dropout:
            self.dropout = nn.Dropout()
        self.elu2 = nn.ELU(inplace=True)

    def forward(self, x, m):
        residual = x
        out = self.conv1(x)
        out = self.elu1(out)
        out = self.conv2(out)
        _, _, h, w = out.size()
        out = self.pcn(out, F.interpolate(m, (h, w), mode="nearest"))
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out = out + residual
        out = self.elu2(out)
        return out

# ==================== HINT Attention 简化集成 ============================


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight[..., None, None] + self.bias[..., None, None]

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            WithBias_LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            WithBias_LayerNorm(dim),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        y = self.avg_pool(x)
        y = self.conv(y)
        y = self.upsample(y)

        out = out * y
        return self.project_out(out)

if __name__ == '__main__':
    pcb = PCBlock(channels_in=3, channels_out=32, kernel_size=5, stride=1, padding=2)
    inp = torch.rand((4, 3, 256, 256))
    mask = torch.rand((4, 1, 256, 256))
    out = pcb(inp, mask)
    print(out.size())
