import torch.nn as nn
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import resnet50


def train( model, train_loader, optimizer, epochs=100):
        device = next(model.parameters()).device
        ce_loss = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for i,(inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                # 计算原始分类损失
                loss_ce = ce_loss(outputs, targets)

                # 组合最终损失
                total_loss = loss_ce

                # 反向传播
                total_loss.backward()
                optimizer.step()
                if i%10==0:
                    _, predicted = torch.max(outputs, 1)
                    correct = (predicted == targets).sum().item()
                    accuracy = correct / targets.size(0)
                    print(f"Epoch {epoch + 1} | Loss: {total_loss.item():.4f}| Acc: {accuracy:.2f}")

            # 每轮结束后保存模型
            torch.save(model.state_dict(), "model_imagenet.pth")
# 初始化基础模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(pretrained=True).to(device)

transform = transforms.Compose([
    transforms.Resize(256),          # Resize the smaller edge to 256
    transforms.CenterCrop(224),      # Crop center 224x224
    transforms.ToTensor(),           # Convert to tensor (0-1 range)
    transforms.Normalize(            # Normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(
        ImageFolder('d:/data/imagenet-mini/train', transform=transform),
        indices=range(500)
    ),
    batch_size=64,
    shuffle=True,
)


# 配置优化器
optimizer = torch.optim.Adam([
   {'params': model.parameters(), 'lr': 1e-4}
])

train(model=model,train_loader=train_loader,optimizer=optimizer,epochs=3)