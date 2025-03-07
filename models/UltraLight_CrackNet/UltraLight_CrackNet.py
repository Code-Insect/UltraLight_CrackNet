import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
from mamba_ssm import Mamba
from torch import Tensor
from einops import rearrange, repeat
from typing import Sequence, Type, Optional
# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SDI(nn.Module):
    def __init__(self, sdi_channel, skip_channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(sdi_channel, sdi_channel, kernel_size=3, stride=1, padding=1) for _ in range(5)])
        self.skip_conv1x1 = nn.Conv2d(sdi_channel, skip_channel, kernel_size=1)

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                  mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)

        return self.skip_conv1x1(ans)


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class PLVMBlock(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = RMSNorm(input_dim)
        self.MSRVSSBlock = SS2D(
            d_model=input_dim // 4,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.Linear = nn.Linear(input_dim, output_dim)


    def forward(self, x):  # B, C, H, W
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        x = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        x_norm = self.norm(x)  # RMSNorm
        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=-1)  # Channel Split
        x_1 = self.MSRVSSBlock(x1)
        x_2 = self.MSRVSSBlock(x2)
        x_3 = self.MSRVSSBlock(x3)
        x_4 = self.MSRVSSBlock(x4)

        x_concat = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
        x_concat = self.norm(x_concat)  # RMSNorm
        out = self.Linear(x_concat)
        out = out.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        return out


class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4),
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)

        return att1, att2, att3, att4, att5


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]


class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5

        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_


class MSConv(nn.Module):
    def __init__(self, dim: int, kernel_sizes: Sequence[int] = (1, 3, 5)) -> None:
        super(MSConv, self).__init__()
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
            for kernel_size in kernel_sizes
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # B, H, W, 4C
        x = x.permute(0, 3, 1, 2).contiguous()  # B 4C H W

        convs_out = x + sum([conv(x) for conv in self.dw_convs])  # multi-scale conv
        out = self.norm(convs_out.permute(0, 2, 3, 1).contiguous())  # B H W 4C
        return out


class MSFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.SiLU,
        drop: float = 0.,
        kernel_sizes: Sequence[int] = (1, 3, 5),
    ) -> None:
        super(MSFFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = nn.Linear

        self.in_linear = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.multiscale_conv = MSConv(hidden_features, kernel_sizes=kernel_sizes)
        self.out_linear = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # B, H, W, C
        x = self.in_linear(x)  # B, H, W, 4C
        x = self.act(x)
        x = self.drop(x)
        x = self.multiscale_conv(x)
        x = self.out_linear(x)
        x = self.drop(x)
        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto",
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model
        self.d_conv = d_conv
        self.expand = expand
        if self.expand is None:
            self.expand = 2

        if 'constrain_ss2d_expand' in kwargs and kwargs['constrain_ss2d_expand']:
            if d_model >= 384:
                self.expand = 1

        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.no_act_branch = False
        if self.no_act_branch:
            self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        else:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.bi_scan = kwargs['bi_scan'] if 'bi_scan' in kwargs else None

        if not self.bi_scan:
            self.x_proj = (
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            )
            self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
            del self.x_proj

            self.dt_projs = (
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
            )
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
            del self.dt_projs

            self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
            self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        else:
            self.x_proj = (
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            )
            self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=2, N, inner)
            del self.x_proj

            self.dt_projs = (
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
            )
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=2, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=2, inner)
            del self.dt_projs

            self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=2, D, N)
            self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=2, D, N)

        self.forward_core = self.forward_corev2

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.MSFFN = MSFFN(in_features=self.d_inner, hidden_features=4 * self.d_inner, out_features=d_model)
        self.skip_scale = nn.Parameter(torch.ones(1))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    # pixmamba seamamba
    def forward_corev2(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W

        if self.bi_scan:
            K = 2
        else:
            K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)

        if self.bi_scan:
            if self.bi_scan == 'xs':
                xs = torch.stack([x.view(B, -1, L), torch.flip(x.view(B, -1, L), dims=[-1])], dim=1)
            else:
                xs = x_hwwh
        else:
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)

        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        D, N = self.A_logs.shape

        out_y = []
        for i in range(K):
            yi = self.selective_scan(
                xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i],
                As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                delta_bias=dt_projs_bias.view(K, -1)[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)

        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        if self.bi_scan:
            if self.bi_scan == 'xs':
                inv_y = torch.flip(out_y[:, 1], dims=[-1]).view(B, -1, L)
                y = out_y[:, 0] + inv_y
            else:
                wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
                # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
                y = out_y[:, 0] + wh_y

            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y

            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        y = self.out_norm(y).to(x.dtype)

        return y

    def forward(self, x: torch.Tensor, **kwargs):  # B, H, W, C
        residual = x

        if self.no_act_branch:
            x = self.in_proj(x)
        else:
            xz = self.in_proj(x)
            x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()  # (b, d, h, w)
        x = self.act(self.conv2d(x))  # (b, d, h, w)  DWConv -> SiLU
        y = self.forward_core(x)
        y = self.out_norm(y)
        if not self.no_act_branch:
            y = y * F.silu(z)  # B, H, W, C

        out = self.MSFFN(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out + self.skip_scale * residual


class UltraLight_CrackNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
                 split_att='fc', bridge=True):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            PLVMBlock(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.encoder5 = nn.Sequential(
            PLVMBlock(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.encoder6 = nn.Sequential(
            PLVMBlock(input_dim=c_list[4], output_dim=c_list[5])
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')


        sdi_channel = c_list[0]
        self.sdi_1 = SDI(sdi_channel, c_list[0])
        self.sdi_2 = SDI(sdi_channel, c_list[1])
        self.sdi_3 = SDI(sdi_channel, c_list[2])
        self.sdi_4 = SDI(sdi_channel, c_list[3])
        self.sdi_5 = SDI(sdi_channel, c_list[4])

        self.conv1x1_2 = BasicConv2d(c_list[1], sdi_channel, 1)
        self.conv1x1_3 = BasicConv2d(c_list[2], sdi_channel, 1)
        self.conv1x1_4 = BasicConv2d(c_list[3], sdi_channel, 1)
        self.conv1x1_5 = BasicConv2d(c_list[4], sdi_channel, 1)


        self.decoder1 = nn.Sequential(
            PLVMBlock(input_dim=c_list[5], output_dim=c_list[4])
        )
        self.decoder2 = nn.Sequential(
            PLVMBlock(input_dim=2 * c_list[4], output_dim=c_list[3])
        )
        self.decoder3 = nn.Sequential(
            PLVMBlock(input_dim=2 * c_list[3], output_dim=c_list[2])
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(2 * c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(2 * c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.ebn6 = nn.GroupNorm(4, c_list[5])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(2 * c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(self.ebn1(self.encoder1(x)))
        t1 = out  # b, c0, H, W
        out = F.max_pool2d(out, 2, 2)

        out = F.gelu(self.ebn2(self.encoder2(out)))
        t2 = out  # b, c1, H/2, W/2
        out = F.max_pool2d(out, 2, 2)

        out = F.gelu(self.ebn3(self.encoder3(out)))
        t3 = out  # b, c2, H/4, W/4
        out = F.max_pool2d(out, 2, 2)

        out = F.gelu(self.ebn4(self.encoder4(out)))
        t4 = out  # b, c3, H/8, W/8
        out = F.max_pool2d(out, 2, 2)

        out = F.gelu(self.ebn5(self.encoder5(out)))
        t5 = out  # b, c4, H/16, W/16
        out = F.max_pool2d(out, 2, 2)

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)


        t2 = self.conv1x1_2(t2)  # b, 8, 256, 256
        t3 = self.conv1x1_3(t3)  # b, 8, 128, 128
        t4 = self.conv1x1_4(t4)  # b, 8, 64, 64
        t5 = self.conv1x1_5(t5)  # b, 8, 32, 32

        t1_1 = self.sdi_1([t1, t2, t3, t4, t5], t1)  # b, 8, 512, 512
        t2_1 = self.sdi_2([t1, t2, t3, t4, t5], t2)  # b, 16, 256, 256
        t3_1 = self.sdi_3([t1, t2, t3, t4, t5], t3)  # b, 24, 128, 128
        t4_1 = self.sdi_4([t1, t2, t3, t4, t5], t4)  # b, 32, 64, 64
        t5_1 = self.sdi_5([t1, t2, t3, t4, t5], t5)  # b, 48, 32, 32


        out = F.gelu(self.ebn6(self.encoder6(out)))  # b, c5, H/32, W/32

        out5 = F.interpolate(F.gelu(self.dbn1(self.decoder1(out))), scale_factor=(2, 2), mode='bilinear', align_corners=True)  # b, c4, H/16, W/16
        # out5 = torch.add(out5, t5_1)  # b, c4, H/16, W/16
        out5 = torch.cat([out5, t5_1], dim=1)  # b, 2 * c4, H/16, W/16

        out4 = F.interpolate(F.gelu(self.dbn2(self.decoder2(out5))), scale_factor=(2, 2), mode='bilinear', align_corners=True)  # b, c3, H/8, W/8
        # out4 = torch.add(out4, t4_1)  # b, c3, H/8, W/8
        out4 = torch.cat([out4, t4_1], dim=1)  # b, 2 * c3, H/8, W/8

        out3 = F.interpolate(F.gelu(self.dbn3(self.decoder3(out4))), scale_factor=(2, 2), mode='bilinear', align_corners=True)  # b, c2, H/4, W/4
        # out3 = torch.add(out3, t3_1)  # b, c2, H/4, W/4
        out3 = torch.cat([out3, t3_1], dim=1)  # b, 2 * c2, H/4, W/4

        out2 = F.interpolate(F.gelu(self.dbn4(self.decoder4(out3))), scale_factor=(2, 2), mode='bilinear', align_corners=True)  # b, c1, H/2, W/2
        # out2 = torch.add(out2, t2_1)  # b, c1, H/2, W/2
        out2 = torch.cat([out2, t2_1], dim=1)  # b, 2 * c1, H/2, W/2

        out1 = F.interpolate(F.gelu(self.dbn5(self.decoder5(out2))), scale_factor=(2, 2), mode='bilinear', align_corners=True)  # b, c0, H, W
        # out1 = torch.add(out1, t1_1)  # b, c0, H, W
        out1 = torch.cat([out1, t1_1], dim=1)  # b, 2 * c0, H, W

        out0 = self.final(out1)  # b, num_class, H, W

        return out0




from thop import profile
def cal_params_flops(model, size):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print('flops: %.2f G' % (flops/1e9))
    print('params: %.2f M' % (params/1e6))

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2f M" % (total/1e6))


if __name__ == '__main__':
    model = UltraLight_CrackNet(num_classes=1,
                                   input_channels=3,
                                   c_list=[8,16,24,32,48,64],  # 8, 16, 32, 64, 128, 256  # 16, 32, 64, 128, 256, 512
                                   split_att='fc',
                                   bridge=True, ).cuda()
    x = torch.rand((4, 3, 384, 384)).cuda()
    y = model(x)
    print(y.size())
    cal_params_flops(model, 384)