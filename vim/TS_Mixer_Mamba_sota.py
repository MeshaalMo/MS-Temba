import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class linear_layer(nn.Module):
    #
    def __init__(self, input_dim=2048, embed_dim=576):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # x = x.transpose(1, 2)
        x = self.proj(x)
        return x


# def resize(input,
#            size=None,
#            scale_factor=None,
#            mode='nearest',
#            align_corners=None):

#     if isinstance(size, torch.Size):
#         size = tuple(int(x) for x in size)
#     return F.interpolate(input, size, scale_factor, mode, align_corners)


class Temporal_Mixer(nn.Module):
    def __init__(self, inter_channels, embedding_dim):
        super().__init__()
        c1_in_channels, c2_in_channels, c3_in_channels = inter_channels

        self.linear_f1 = linear_layer(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.linear_f2 = linear_layer(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_f3 = linear_layer(input_dim=c3_in_channels, embed_dim=embedding_dim)

    def forward(self, x):

        f1, f2, f3 = x

        _f1 = self.linear_f1(f1)
        _f2 = self.linear_f2(f2)
        _f3 = self.linear_f3(f3)

        concat_feature = torch.cat([_f1, _f2, _f3], dim=1)

        return concat_feature

class Temporal_Interaction_Block(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = None or head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # elif isinstance(m, nn.Conv1d):
        #     fan_out = m.kernel_size[0] * m.out_channels
        #     fan_out //= m.groups
        #     m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        #     if m.bias is not None:
        #         m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B, 8, N, C//8
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 2, B, 8, N, C//8
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x