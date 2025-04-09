import itertools
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import resnet50
from skimage.transform import resize
import torch.nn.functional as F


class DMIEstimator(nn.Module):

    def __init__(self, model, local_layers, global_layer, dim):
        super().__init__()
        self.local_layers = local_layers
        self.global_layer = global_layer
        self.local_features = []
        self.model = model

        # 计算本地层的总输入通道数
        combined_channels = sum([layer.in_channels for layer in local_layers])

        # 定义掩码生成模块，适配本地层通道数
        self.mask_block = nn.Sequential(
            nn.Conv2d(combined_channels, combined_channels, kernel_size=3, padding=1),
            nn.Conv2d(combined_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 计算MLP的输入维度（本地通道数 + 全局维度）
        mlp_input_dim = local_layers[-1].in_channels + dim
        hidden_dim = 256  # MLP隐藏层大小
        self.mlp_raw = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def register(self):
        def layer_hook(m, input):
            if self.hook:
                if len(self.local_features) == len(self.local_layers)-1:
                    self.local_features = []
                self.local_features.append(input[0].detach())
            else:
                pass

        def local_forward_pre_hook(m, input):
            if self.hook:
                # 捕获原始本地特征用于互信息计算
                if len(self.local_layers) > 1:
                    self.raw_local_features = torch.cat([
                        F.adaptive_avg_pool2d(self.local_features[i], input[0].shape[2:]) for i in range(len (self.local_layers)-1)
                    ], dim=1)
                    self.raw_local_feature = input[0].detach()
                    self.raw_local_features = torch.cat([self.raw_local_features, input[0].detach()], dim=1)
                else:
                    self.raw_local_features = input[0].detach()
                    self.raw_local_feature = self.raw_local_features

                # 生成并应用掩码，返回处理后的输入
                self.mask_value = self.mask_block(self.raw_local_features)
                masked_input = self.mask_value.expand_as(input[0]) * input[0]
                self.mask_local_feature = masked_input
                return (masked_input,)
            else:
                return input

        def global_forward_hook(m, input, output):
            if self.hook:
                # 存储全局特征
                self.mask_global_feature = output.detach()

                # 计算原始和掩码后的互信息
                self.raw_mi = self.calculate_mi(self.raw_local_feature.detach(), self.raw_global_feature, is_raw=True)
                self.mask_mi = self.calculate_mi(self.mask_local_feature.detach(), self.raw_global_feature, is_raw=False)
            else:
                self.raw_global_feature = output.detach()
                return output

        [self.local_layers[i].register_forward_pre_hook(layer_hook) for i in range(len(self.local_layers)-1)]
        self.local_layers[-1].register_forward_pre_hook(local_forward_pre_hook)
        self.global_layer.register_forward_hook(global_forward_hook)


    def calculate_mi(self, local_feature, global_feature, is_raw=True):
        n = local_feature.shape[0]
        h = local_feature.shape[2]
        w = local_feature.shape[3]
        nhw = local_feature.shape[0] * local_feature.shape[2] * local_feature.shape[3]
        hw = local_feature.shape[2] * local_feature.shape[3]
        local_feature = local_feature.permute(0, 2, 3, 1)
        global_feature = global_feature[:, :, 0, 0]
        mi = torch.zeros(n, h, w)

        for i in range(n):
            combined = torch.cat([global_feature[i:i+1, :].repeat(nhw, 1), local_feature.reshape(nhw, -1)], -1)
            scores = self.mlp_raw(combined).squeeze()  # 计算得分

            max_score = scores.max()  # 用于数值稳定性
            exp_scores = torch.exp(scores - max_score)

            # 计算负样本平均值
            neg_indices = list(set(range(nhw)) - set(range(hw * i, hw * (i + 1))))
            neg_mean = exp_scores[neg_indices].mean() + 1e-10

            log_term = (scores[hw * i:hw * (i + 1)] - max_score) - torch.log(neg_mean)
            mi[i, :] = log_term.reshape((h, w))

        if is_raw:
            self.local_raw_mi = mi
        else:
            self.local_mask_mi = mi
        return mi.mean()


    def visual_heatmap(self,epoch, input_t: torch.Tensor, target_t: torch.Tensor, img_num=4):
        self.model.eval()
        with torch.no_grad():
            self.model(input_t)

        # 反归一化生成原始图像
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(input_t.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(input_t.device)
        original_img = input_t * std + mean
        original_img = original_img.clamp(0, 1)

        # 生成原始互信息热图
        if input_t.shape[0] > 1:
            mi_raw_heatmaps = np.zeros((input_t.shape[0], input_t.shape[2], input_t.shape[3]))
            for i in range(input_t.shape[0]):
                htensor = self.local_raw_mi[i].cpu().numpy()
                hmap = resize(htensor, input_t.shape[2:])
                mi_raw_heatmaps[i] = hmap
        else:
            htensor = self.local_raw_mi[0].cpu().numpy()
            hmap = resize(htensor, input_t.shape[2:])
            mi_raw_heatmaps = hmap

        # 生成掩码互信息热图
        if input_t.shape[0] > 1:
            mi_mask_heatmaps = np.zeros((input_t.shape[0], input_t.shape[2], input_t.shape[3]))
            for i in range(input_t.shape[0]):
                htensor = self.local_mask_mi[i].cpu().numpy()
                hmap = resize(htensor, input_t.shape[2:])
                mi_mask_heatmaps[i] = hmap
        else:
            htensor = self.local_mask_mi[0].cpu().numpy()
            hmap = resize(htensor, input_t.shape[2:])
            mi_mask_heatmaps = hmap

        # 可视化热图
        results = []
        for i in range(min(4, input_t.shape[0])):
            blend_fig = plt.figure()
            plt.imshow(original_img[i].permute(1, 2, 0).cpu().numpy())

            # 归一化原始互信息热图
            mi_data = mi_raw_heatmaps[i] if input_t.shape[0] > 1 else mi_raw_heatmaps
            mi_data = (mi_data - mi_data.min()) / (mi_data.max() - mi_data.min())
            plt.imshow(mi_data, cmap='jet', alpha=0.5)
            plt.axis('off')
            plt.savefig(f'output/raw_mi_heatmap_{epoch}_{i}.png')  # 保存图像
            plt.show()  # 关闭图，避免内存占用

            blend_fig = plt.figure()
            plt.imshow(original_img[i].permute(1, 2, 0).cpu().numpy())

            # 归一化掩码互信息热图
            mi_data = mi_mask_heatmaps[i] if input_t.shape[0] > 1 else mi_mask_heatmaps
            mi_data = (mi_data - mi_data.min()) / (mi_data.max() - mi_data.min())
            plt.imshow(mi_data, cmap='jet', alpha=0.5)
            plt.savefig(f'output/mask_mi_heatmap_{epoch}_{i}.png')
            plt.show()

            # 归一化差值热图
            mi_data = mi_raw_heatmaps[i] - mi_mask_heatmaps[i] if input_t.shape[0] > 1 else mi_raw_heatmaps - mi_mask_heatmaps
            mi_data = (mi_data - mi_data.min()) / (mi_data.max() - mi_data.min())

            plt.imshow(mi_data, cmap='jet', alpha=0.5)
            plt.axis('off')
            plt.savefig(f'output/diff_mi_heatmap_{epoch}_{i}.png')
            plt.show()

    def train_method(self, model, train_loader, optimizer, epochs=100, mi_weight=10, mask_reg_weight=1):
        """训练方法，结合互信息和分类损失"""
        device = next(model.parameters()).device
        ce_loss = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                with torch.no_grad():
                    self.hook = False
                    model(inputs)
                    self.hook = True

                outputs = model(inputs)

                # 计算分类损失
                loss_ce = ce_loss(outputs, targets)

                # 计算互信息损失（最大化互信息）
                loss_mi = -(self.raw_mi + self.mask_mi)

                # 组合总损失
                total_loss = mi_weight * loss_mi + loss_ce + mask_reg_weight * self.mask_value.mean()

                total_loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print(f"Epoch {epoch + 1} | Loss: {total_loss.item():.4f} (CE: {loss_ce.item():.4f}, MI: {loss_mi.item():.4f}, mask: {self.mask_value.mean().item():.4f})")

            if epoch % 5 == 0:
                self.visual_heatmap(epoch,inputs, targets, img_num=2)

        # 保存模型参数
        torch.save(self.state_dict(), "model.pth")


# 设置设备和模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(pretrained=True).to(device)
model.load_state_dict(torch.load('model_imagenet.pth'))

# 配置DMI模块
global_layer = model.avgpool
dmi_module = DMIEstimator(
    model=model,
    local_layers=[model.layer4[2].conv3],
    global_layer=global_layer,
    dim=2048  # ResNet-50全局特征维度
).to(device)

# 注册前向钩子
dmi_module.register()

# 定义数据预处理
imagenet_path = 'd:/data/imagenet-mini/train'
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(
        ImageFolder(imagenet_path, transform=transform),
        indices=range(100)
    ),
    batch_size=16,
    shuffle=True,
)

# 配置优化器
optimizer = torch.optim.Adam([
    {'params': dmi_module.mask_block.parameters(), 'lr': 1e-3},
    {'params': dmi_module.mlp_raw.parameters(), 'lr': 1e-3},
])

# 开始训练
dmi_module.train_method(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    epochs=100,
    mi_weight=1,
    mask_reg_weight=0.5
)