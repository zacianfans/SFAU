import torch
import torch.nn as nn
import torch.nn.functional as fun
from model.BasicBlock import *
from torchsummary import summary
from model.Block import Ddnf, SpaBlock, ChannelAttention

def sim(q, k, kernel_size=5, scale=2):
    B, H, W, C = k.shape
    q = q.view(B, H, scale, W, scale, C)
    k = F.unfold(k.permute(0, 3, 1, 2), kernel_size=kernel_size, padding=kernel_size // 2).reshape(
            B, C, kernel_size ** 2, H, W)
    return torch.einsum('ijklmn,inojl->ijklmo', q, k).reshape(
            B, scale * H, scale * W, kernel_size ** 2).contiguous()


def atn(attn, x, kernel_size=5, scale=2):
    B, H, W, C = x.shape
    attn = attn.view(B, H, scale, W, scale, kernel_size ** 2)
    x = F.unfold(x.permute(0, 3, 1, 2), kernel_size=kernel_size, stride=1, padding=kernel_size // 2).view(
        B, C, kernel_size ** 2, H, W)
    return torch.einsum('ijklmn,ionjl->ijklmo', attn, x).contiguous().view(B, H * scale, W * scale, C)

class DDNF(nn.Module):

    def __init__(self, in_nc):
        super(DDNF, self).__init__()
        self.spatial_process = SpaBlock(in_nc)  # 空间处理块
        self.frequency_process = Ddnf(in_nc)  # 频率处理块
        self.cat = nn.Conv2d(2 * in_nc, in_nc//2, 1, 1, 0)  # 用于合并两个域信息的1x1卷积

    # 前向传播函数
    def forward(self, x):
        _, _, H, W = x.shape  # 获取输入尺寸

        x_freq = self.frequency_process(x)  # 频率域处理  torch.Size([4, 32, 64, 64])
        x = self.spatial_process(x)  # 空间域处理    torch.Size([4, 32, 64, 64])

        xcat = torch.cat([x, x_freq], 1)  # 在通道维度上合并
        x_out = self.cat(xcat)  # 应用1x1卷积
        return x_out  # 返回最终输出

class SFAU(nn.Module):
    def __init__(self, y_channels, x_channels, embedding_dim=64, k_up=5, scale=4):
        super(SFAU, self).__init__()
        self.scale = scale
        self.k_up = k_up

        # ms 图像特征提取
        # self.ms_conv = MSFF(x_channels, 16)

        # gate
        self.gate = nn.Conv2d(x_channels, x_channels, 1)

        # ms与pan的特征融合
        self.pan_conv1 = nn.Sequential(
        nn.Conv2d(y_channels + x_channels, 16, 1),
        nn.Conv2d(16, 16, 5, padding=2))
        self.pan_conv2 = DDNF(16) # 输出通道减半
        self.refine = ChannelAttention(8, 4)
        self.pan_conv3 = nn.Conv2d(8, x_channels, kernel_size=1)
        # PAN 图像特征提取
        # self.pan_conv = nn.Sequential(
        #     nn.Conv2d(5, 32, kernel_size=3, padding=1),
        #     nn.Conv2d(32, 32, kernel_size=5, padding=2),
        #     RefineWithResidual(32, 4),
        #     nn.Conv2d(32, 4, kernel_size=3, padding=1)
        # )


        self.norm_y = nn.LayerNorm(x_channels)
        self.norm_x = nn.LayerNorm(x_channels)

        self.q = nn.Linear(x_channels, embedding_dim)
        self.k = nn.Linear(x_channels, embedding_dim)

    def forward(self, x, y): # ms, pan
        up_x = fun.interpolate(x, scale_factor=(self.scale, self.scale), mode='bicubic')
        gate = self.gate(up_x)
        # 使用 sigmoid 生成门控值（0 到 1 之间）
        gate = torch.sigmoid(gate)  # gate 尺寸仍为 4x128x128
        y = self.pan_conv1(torch.cat([y, up_x], dim=1))
        y = self.pan_conv2(y)
        y = self.refine(y)
        enc_feature = self.pan_conv3(y)
        y = enc_feature.permute(0, 2, 3, 1).contiguous()
        x = x.permute(0, 2, 3, 1).contiguous()
        y = self.norm_y(y)
        x_ = self.norm_x(x)

        q = self.q(y)
        k = self.k(x_)

        sapa_output = self.attention(q, k, x).permute(0, 3, 1, 2).contiguous() # 输出通道为4，与这里的x一致

        final_output = gate * sapa_output + (1 - gate) * enc_feature
        return final_output, sapa_output, enc_feature

    def attention(self, q, k, v):
        attn = F.softmax(sim(q, k, self.k_up, self.scale), dim=-1)
        return atn(attn, v, self.k_up, self.scale)


if __name__ == '__main__':
    pan = torch.ones((32, 1, 128, 128))
    ms = torch.ones((32, 4, 32, 32))
    model = SFAU(1, 4)
    output1, output2, _ = model.forward(ms, pan)
    summary(model, input_size=[(4, 32, 32), (1, 128, 128)], device='cpu')
    print(output1.shape, output2.shape)