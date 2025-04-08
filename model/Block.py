import torch
import torch.nn.functional as F
from torch import nn

class MSFF(nn.Module):
    def __init__(self, inchannel, mid_channel):
        # 调用父类的构造函数
        super(MSFF, self).__init__()
        # 定义第一个卷积序列
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),  # 1x1卷积
            nn.BatchNorm2d(inchannel),  # 批归一化
            nn.ReLU(inplace=True)  # ReLU激活
        )
        # 定义第二个卷积序列，使用3x3卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),  # 1x1卷积降维
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 3, stride=1, padding=1, bias=False),  # 3x3卷积
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),  # 1x1卷积升维
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        # 定义第三个卷积序列，使用5x5卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 5, stride=1, padding=2, bias=False),  # 5x5卷积
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        # 定义第四个卷积序列，使用7x7卷积
        self.conv4 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 7, stride=1, padding=3, bias=False),  # 7x7卷积
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        # 定义混合卷积序列
        self.convmix = nn.Sequential(
            nn.Conv2d(4 * inchannel, inchannel, 1, stride=1, bias=False),  # 1x1卷积降维
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, inchannel, 3, stride=1, padding=1, bias=False),  # 3x3卷积
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 通过不同的卷积序列
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        # 在通道维度上拼接
        x_f = torch.cat([x1, x2, x3, x4], dim=1)
        # 通过混合卷积序列
        out = self.convmix(x_f)

        # 返回输出
        return out


class DEAM(nn.Module):
    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(DEAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in  # 降低通道数用于计算注意力
        self.activation = activation
        self.ds = ds  # 下采样因子
        self.pool = nn.AvgPool2d(self.ds)  # 平均池化用于下采样
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # 查询卷积
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # 键卷积
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # 值卷积
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习参数用于调整注意力强度
        self.softmax = nn.Softmax(dim=-1)  # softmax用于计算注意力

    def forward(self, input, diff):
        """
            inputs :
                x : 输入特征图 (B X C X W X H)
            returns :
                out : 自注意力值与输入特征相加
                attention: 注意力矩阵 B X N X N (N 是宽度*高度)
        """
        diff = self.pool(diff)  # 对差异图进行下采样
        m_batchsize, C, width, height = diff.size()
        # 计算查询向量
        proj_query = self.query_conv(diff).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        # 计算键向量
        proj_key = self.key_conv(diff).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        # 计算注意力能量
        energy = torch.bmm(proj_query, proj_key)  # 矩阵乘法
        energy = (self.key_channel ** -.5) * energy  # 标准化
        # 计算注意力矩阵
        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        x = self.pool(input)  # 对输入进行下采样
        # 计算值向量
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        # 注意力加权值
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        # 上采样到原始尺寸
        out = F.interpolate(out, [width * self.ds, height * self.ds])
        out = out + input  # 与输入相加

        return out

import torch
import torch.nn as nn

'''
    论文地址：https://openaccess.thecvf.com/content/CVPR2024/html/Cong_A_Semi-supervised_Nighttime_Dehazing_Baseline_with_Spatial-Frequency_Aware_and_Realistic_CVPR_2024_paper.html
    论文题目：A Semi-supervised Nighttime Dehazing Baseline with Spatial-Frequency Aware and Realistic Brightness Constraint（CVPR 2024）
    中文题目：具有空间频率感知和现实亮度约束的半监督夜间除雾基线模型
    讲解视频：https://www.bilibili.com/video/BV1pySxYkEi1/
      本地感知的双向非线性映射（Bidomain Local Perception and Nonlinear Mapping，BLPNM）：
                    Bidomain 非线性映射（BNM）是一种用于计算窗口注意力的方法，但计算窗口注意力不具有非线性表示能力。
                    因此，对频率域信息和空间域信息实现非线性映射，并在此基础上进行全局感知的自注意力计算。
'''
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:  # 如果同时使用偏置和归一化，则不使用偏置
            bias = False
        padding = kernel_size // 2  # 根据卷积核大小设置填充
        layers = list()  # 创建一个层列表
        if transpose:  # 如果是转置卷积
            padding = kernel_size // 2 - 1  # 转置卷积时调整填充
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))  # 添加转置卷积层
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))  # 添加普通卷积层
        if norm:  # 如果使用批归一化
            layers.append(nn.BatchNorm2d(out_channel))  # 添加批归一化层
        if relu:  # 如果使用激活函数
            layers.append(nn.GELU())  # 添加GELU激活函数
        self.main = nn.Sequential(*layers)  # 将所有层组成序列

    # 前向传播函数
    def forward(self, x):
        return self.main(x)  # 应用前面定义的所有层

# 定义空间块
class SpaBlock(nn.Module):
    # 初始化函数
    def __init__(self, nc):
        super(SpaBlock, self).__init__()  # 调用父类初始化方法
        in_channel = nc
        out_channel = nc
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)  # 第一层卷积
        self.trans_layer = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)  # 过渡层
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)  # 第二层卷积
    def forward(self, x):
        out = self.conv1(x)  # 经过第一层卷积
        out = self.trans_layer(out)  # 经过过渡层
        out = self.conv2(out)  # 经过第二层卷积
        return out + x  # 加上原始输入形成残差连接

# 定义SE（Squeeze-and-Excitation）层
class SELayer(nn.Module):
    # 初始化函数
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()  # 调用父类初始化方法
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 线性变换降维
            nn.ReLU(inplace=True),  # ReLU激活
            nn.Linear(channel // reduction, channel, bias=False),  # 线性变换升维
            nn.Sigmoid()  # Sigmoid激活
        )

    # 前向传播函数
    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入尺寸
        y = self.avg_pool(x).view(b, c)  # 全局平均池化后展平
        y = self.fc(y).view(b, c, 1, 1)  # 经过线性变换并恢复形状
        return x * y.expand_as(x)  # 对输入进行加权


class Ddnf(nn.Module):
    # 初始化函数
    def __init__(self, nc):
        super(Ddnf, self).__init__()  # 调用父类初始化方法
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),  # 1x1卷积处理幅度
            nn.LeakyReLU(0.1, inplace=True),  # LeakyReLU激活
            SELayer(channel=nc),  # SE层
            nn.Conv2d(nc, nc, 1, 1, 0)  # 另一个1x1卷积
        )
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),  # 1x1卷积处理相位
            nn.LeakyReLU(0.1, inplace=True),  # LeakyReLU激活
            SELayer(channel=nc),  # SE层
            nn.Conv2d(nc, nc, 1, 1, 0)  # 另一个1x1卷积
        )

    # 前向传播函数
    def forward(self, x):
        _, _, H, W = x.shape  # 获取输入尺寸

        x_freq = torch.fft.rfft2(x, norm='backward')  # 计算二维实数FFT

        ori_mag = torch.abs(x_freq)  # 计算幅度
        mag = self.processmag(ori_mag)  # 处理幅度
        mag = ori_mag + mag  # 残差连接

        ori_pha = torch.angle(x_freq)  # 计算相位
        pha = self.processpha(ori_pha)  # 处理相位
        pha = ori_pha + pha  # 残差连接

        real = mag * torch.cos(pha)  # 计算实部
        imag = mag * torch.sin(pha)  # 计算虚部

        x_out = torch.complex(real, imag)  # 合成复数输出

        x_freq_spatial = torch.fft.irfft2(x_out, s=(H, W), norm='backward')  # 逆FFT转换回空间域
        return x_freq_spatial  # 返回结果

# 定义双域非线性映射模块
class BidomainNonlinearMapping(nn.Module):

    def __init__(self, in_nc):
        super(BidomainNonlinearMapping, self).__init__()
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


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        res = self.process(x)
        y = self.avg_pool(res)
        z = self.conv_du(y)
        return z * res + x
