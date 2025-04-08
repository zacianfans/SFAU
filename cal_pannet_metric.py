import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataset import MyDataset
from utils.metrics import *
from model.PanNet import PanNet_SFAU, PanNet_PGCU
# 计算模型的各种评价指标
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
batch_size = 32

# prepare dataset & dataloader
data_root = '../data/WV2_data'  # 更换数据集
test_pan = 'test128/pan'
test_ms = 'test128/ms'
test_dataset = MyDataset(data_root, test_ms, test_pan, 'bicubic')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
# Model = PanNet_CARAFE(4).to(device)  # 实例化模型类
Model = PanNet_SFAU(1, 4).to(device)
# Model = PanNet_SANM_A(1, 4).to(device)
# Model = PanNet_DySample(1, 4).to(device)
# Model = PanNet_TConv(1, 4).to(device)
# 加载模型权重
state_dict = torch.load('run/PanNet_CARAFE/PanNet_CARAFE_WV2_20250102-180500/PanNet_CARAFE_best_test_loss.pth', map_location=device)  # 加载模型的状态字典
Model.load_state_dict(state_dict)  # 将状态字典加载到模型中

Model.to(device)  # 将模型转移到设备（GPU或CPU）
Model.eval()  # 设置为评估模式

# Initialize metric accumulators
g_metrics_sum = np.zeros(6)
num_images = 0

for label_batch, pan_batch, lrms_batch, up_ms_batch, hpan_batch, hlrms_batch in tqdm(test_loader):
    label_batch = torch.Tensor(label_batch).to(device).float()
    pan_batch = torch.Tensor(pan_batch).to(device).float()
    lrms_batch = torch.Tensor(lrms_batch).to(device).float()
    hpan_batch = torch.Tensor(hpan_batch).to(device).float()
    hlrms_batch = torch.Tensor(hlrms_batch).to(device).float()

    batch_size = label_batch.size(0)
    for i in range(batch_size):
        label = label_batch[i]
        pan = pan_batch[i]
        lrms = lrms_batch[i]
        hpan = hpan_batch[i]
        hlrms = hlrms_batch[i]

        # PanNet_GAU (仅计算pannet_pgcu的指标)
        with torch.no_grad():
            out, up_ms = Model(pan.unsqueeze(0), lrms.unsqueeze(0), hpan.unsqueeze(0))  # 执行推理
        g_metrics = ref_evaluate(out.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0), label.cpu().detach().numpy().transpose(1, 2, 0))
        g_metrics_sum += np.array(g_metrics)

        num_images += 1

# Calculate average metrics
g_metrics_avg = g_metrics_sum / num_images

# Print results
print(num_images)
print("PanNet_GAU Average Metrics:")
print(
    f"PSNR: {g_metrics_avg[0]:.4f}, SSIM: {g_metrics_avg[1]:.4f}, SAM: {g_metrics_avg[2]:.4f}, ERGAS: {g_metrics_avg[3]:.4f}, SCC: {g_metrics_avg[4]:.4f}, Q: {g_metrics_avg[5]:.4f}")
