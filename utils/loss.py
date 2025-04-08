import torch
import torch.nn as nn
import torchvision.models as models


class VGG19Features(nn.Module):
    def __init__(self):
        super(VGG19Features, self).__init__()

        # 选择所需的卷积层（对应conv1_2, conv2_2, conv3_3）
        self.feature_layers = [2, 7, 14]

        # 加载预训练的VGG19模型
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

        # 截取到所需的层并创建新的Sequential模型
        self.model = nn.Sequential(*list(vgg19.features.children())[:self.feature_layers[-1] + 1])

        # 冻结参数，不需要梯度更新
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 对输入图像进行归一化，将其范围从 [0, 1] 变换到 [-1, 1]
        x = (x - 0.5) / 0.5
        features = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = VGG19Features()
        self.weights = [1, 0.2, 0.04]  # 不同层次的损失权重
        self.mse_loss = nn.MSELoss()

    def forward(self, fake_img, real_img):
        # 计算生成图像和真实图像的VGG特征
        fake_features = self.vgg(fake_img)
        real_features = self.vgg(real_img)

        # 计算感知损失
        loss = 0
        for i in range(len(fake_features)):
            loss += self.weights[i] * self.mse_loss(fake_features[i], real_features[i].detach())

        return loss


if __name__ == '__main__':
    # 创建VGGPerceptualLoss实例
    perceptual_loss = VGGPerceptualLoss()

    # 随机生成生成图像和真实图像
    fake_img = torch.rand(1, 3, 224, 224)
    real_img = torch.rand(1, 3, 224, 224)

    # 计算感知损失
    loss = perceptual_loss(fake_img, real_img)
    print(loss)