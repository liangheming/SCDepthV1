import torch
from torch import nn
from torch.nn import functional as f
from models.resnet import get_backbone


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_refl=True, bias=True):
        super(BasicConv, self).__init__()
        padding = kernel_size // 2
        padding_mode = "reflect" if use_refl else "zeros"
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, 1, padding,
                              padding_mode=padding_mode, bias=bias)
        self.norm = None if bias else nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x if self.norm is None else self.norm(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_refl=True, bias=True):
        super(ConvBlock, self).__init__()
        self.conv = BasicConv(in_channels, out_channels, kernel_size, use_refl, bias)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), alpha=10.0, beta=0.01):
        super(DepthDecoder, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = [16, 32, 64, 128, 256]

        self.conv_before_up_sample = list()
        self.conv_after_up_sample = list()
        self.conv_for_disp = list()

        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.conv_before_up_sample.append(ConvBlock(num_ch_in, num_ch_out))
            num_ch_in = num_ch_out if i == 0 else num_ch_out + self.num_ch_enc[i - 1]
            self.conv_after_up_sample.append(ConvBlock(num_ch_in, num_ch_out))
        for s in self.scales:
            self.conv_for_disp.append(
                BasicConv(self.num_ch_dec[s], 1)
            )
        self.conv_before_up_sample = nn.ModuleList(self.conv_before_up_sample)
        self.conv_after_up_sample = nn.ModuleList(self.conv_after_up_sample)
        self.conv_for_disp = nn.ModuleList(self.conv_for_disp)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        ret = list()
        x = xs[-1]
        for idx, i in enumerate(range(4, -1, -1)):
            x = self.conv_before_up_sample[idx](x)
            x = f.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = torch.cat([x, xs[i - 1]], dim=1) if i > 0 else x
            x = self.conv_after_up_sample[idx](x)
            if i in self.scales:
                disp = self.alpha * self.sigmoid(self.conv_for_disp[i](x)) + self.beta
                depth = 1.0 / disp
                ret.append(depth)
        return ret[::-1]


class DepthNet(nn.ModuleList):
    def __init__(self, name="resnet18", pretrained=True):
        super(DepthNet, self).__init__()
        self.encoder = get_backbone(name, pretrained)
        self.decoder = DepthDecoder(self.encoder.out_channels)

    def forward(self, x):
        return self.decoder(self.encoder(x))[0]


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc):
        super(PoseDecoder, self).__init__()
        self.convs = nn.Sequential(
            BasicConv(num_ch_enc[-1], 256, 1, use_refl=False, bias=True),
            nn.ReLU(inplace=True),
            BasicConv(256, 256, 3, use_refl=False, bias=True),
            nn.ReLU(inplace=True),
            BasicConv(256, 256, 3, use_refl=False, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 6, 1)
        )

    def forward(self, xs):
        x = self.convs(xs[-1])
        x = x.mean(3).mean(2)
        x = 0.01 * x
        return x


class PoseNet(nn.Module):
    def __init__(self, name="resnet18", pretrained=True):
        super(PoseNet, self).__init__()
        self.encoder = get_backbone(name, pretrained, input_images=2)
        self.decoder = PoseDecoder(self.encoder.out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.decoder(self.encoder(x))


if __name__ == '__main__':
    net = PoseNet()
    inp = torch.randn(size=(4, 6, 128, 128))
    outs = net(inp)
    print(outs.shape)
