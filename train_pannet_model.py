import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import json
import os
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataset import MyDataset
import time
from model.PanNet import PanNet_PGCU, PanNet_SFAU, PanNet_SANM_A # PanNet_SANM_A是用来做消融实验的
# 固定随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法相同
    torch.backends.cudnn.benchmark = False  # 禁用以使其不自动寻找最优卷积算法
seed = 42
set_seed(seed)  # 设置随机种子

# -------------------------------------------全局配置-------------------------------------------
project = 'PanNet'
model_name = 'PanNet_SANM'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
epochs = 500
batch_size = 32
loss = "L1" # MSE L1
g_weight_decay = 1e-5
g_learning_rate = 5e-4 # 5e-4
lamda = 0 # 额外的损失函数的权重
k_up = 5 # 上采样窗口大小
datasets = "WV3"
# ----------------------------------------以上是超参数设置部分----------------------------------------

# 生成时间戳
timestamp = time.strftime("%Y%m%d-%H%M%S") # 使用当前时间戳唯一命名文件夹
run_name = f"{model_name}_{datasets}_{timestamp}"
# 初始化 wandb
wandb.init(project=f'{project}', entity='1136396144', name=run_name)

# 准备数据集和 DataLoader
data_root = f'./data/{datasets}_data'
train_pan = 'train128/pan'
train_ms = 'train128/ms'
test_pan = 'test128/pan'
test_ms = 'test128/ms'
train_dataset = MyDataset(data_root, train_ms, train_pan, 'bicubic')
test_dataset = MyDataset(data_root, test_ms, test_pan, 'bicubic')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# PanNet_GAU 配置
if loss == "MSE":
    g_lossFun = nn.MSELoss()
elif loss == "L1":
    g_lossFun = nn.L1Loss()

# 以下代码在选择不同模型时需要修改
# Model = PanNet_PGCU(4, 128).to(device)
Model = PanNet_SFAU(1, 4).to(device)

# 如果需要加载预训练模型，取消注释以下代码
# state_dict = torch.load('run/run_20241121-160043/PANNet_CARAFE_best_test_loss.pth', map_location=device)  # 加载模型的状态字典
# Model.load_state_dict(state_dict)  # 将状态字典加载到模型中

# 多种优化器可供选择
g_optimizer = optim.Adam(Model.parameters(), lr=g_learning_rate, weight_decay=g_weight_decay)
# g_optimizer = optim.SGD(Model.parameters(), lr=g_learning_rate, weight_decay=g_weight_decay, momentum=0.9)
# 多种学习率调度器可供选择
scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=epochs, eta_min=5e-7)
# scheduler_2 = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=100, gamma=0.3)
# scheduler_2 = optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, mode='min', factor=0.1, patience=10)

# 记录训练和测试损失
g_train_loss = []
g_test_loss = []

# 初始化最佳损失以跟踪最佳模型
best_loss = float('inf')
best_model_state = None

# 创建当前运行的文件夹
run_folder = f'run/{model_name}'
os.makedirs(run_folder, exist_ok=True)


run_path = os.path.join(run_folder, run_name)
os.makedirs(run_path, exist_ok=True)

# 将配置详情保存到 JSON 文件
config = {
    "device": device,
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": g_learning_rate,
    "weight_decay": g_weight_decay,
    "train_dataset": os.path.join(data_root, train_ms),
    "test_dataset": os.path.join(data_root, test_ms),
    "model": model_name,
    "timestamp": timestamp,
    "random_seed": seed,
    "loss": loss,
    "lamda": lamda,
    "k_up": k_up,
}

with open(os.path.join(run_path, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)

for epoch in tqdm(range(epochs), desc="Epochs"):
    # 训练阶段
    g_loss = 0
    Model.train()
    for batch in tqdm(train_loader, desc="Training Batches"):
        label, pan, lrms, up_ms, hpan, hlrms = batch
        label = label.to(device).float()
        pan = pan.to(device).float()
        hpan = hpan.to(device).float()
        lrms = lrms.to(device).float()
        hlrms = hlrms.to(device).float()

        # 前向传播
        out, up_ms = Model.forward(pan, lrms, hpan)
        loss_1 = g_lossFun(out, label)
        # optional: for residual structure
        loss_2 = g_lossFun(up_ms, label)

        loss = loss_1 + loss_2 * lamda
        # 反向传播和优化
        g_optimizer.zero_grad()
        loss.backward()
        g_optimizer.step()

        g_loss += loss_1.item()

    avg_train_loss = g_loss / len(train_loader)
    g_train_loss.append(avg_train_loss)
    print(f'Epoch: {epoch + 1}/{epochs}, {model_name} Train Loss: {avg_train_loss:.6f}')

    # 将训练损失记录到 wandb
    wandb.log({
        'epoch': epoch + 1,
        f'{model_name} train loss': avg_train_loss
    })

    # 每一轮进行一次测试
    g_loss = 0
    Model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Batches"):
            label, pan, lrms, up_ms, hpan, hlrms = batch
            label = label.to(device).float()
            pan = pan.to(device).float()
            lrms = lrms.to(device).float()
            hpan = hpan.to(device).float()
            hlrms = hlrms.to(device).float()

            # 前向传播
            out, up_ms = Model.forward(pan, lrms, hpan)
            loss_1 = g_lossFun(out, label)
            # optional: for residual structure
            # loss_2 = g_lossFun(up_ms, label)

            # loss = loss_1 + loss_2 * 0.1

            g_loss += loss_1.item()

    avg_test_loss = g_loss / len(test_loader)
    g_test_loss.append(avg_test_loss)
    print(f'Epoch: {epoch + 1}/{epochs}, {model_name} Test Loss: {avg_test_loss:.6f}')

    # 将测试损失记录到 wandb
    wandb.log({
        'epoch': epoch + 1,
        f'{model_name} test loss': avg_test_loss
    })

    # 保存具有最低测试损失的模型
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        best_model_state = Model.state_dict()
        torch.save(best_model_state, os.path.join(run_path, f'{model_name}_best_test_loss.pth'))
        print(f"New best model saved at epoch {epoch + 1} with test loss: {avg_test_loss:.6f}")

    # 更新学习率
    scheduler_2.step()

# 完成 wandb 运行
wandb.finish()


