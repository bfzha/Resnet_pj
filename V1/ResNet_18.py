# 导入必要的库
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import progress_bar
# 设置超参数
BATCH_SIZE = 512
NUM_CLASSES = 10
LR = 0.1  # SGD
# LR = 3e-4 # Adam
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1.准备数据集
MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(
            p=0.5
        ),  # 以0.5的概率水平翻转图像，增加数据多样性
        transforms.RandomCrop(
            32, padding=4
        ),  # 32x32的图像，填充4个像素后再裁剪回32x32 模拟图像小范围位移，提高平移不变性
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
        ),  # 随机调整图像的亮度、对比度、饱和度和色调，增强模型对颜色变化的鲁棒性
        transforms.RandomRotation(degrees=9),  # 随机旋转图像，增加模型对旋转的鲁棒性
        transforms.RandomAffine(
            degrees=0, translate=(0.045, 0.036)
        ),  # 随机仿射变换，模拟图像平移
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(
            p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"
        ),  # 随机擦除图像的一部分，模拟遮挡情况，增强模型的鲁棒性
    ]
)

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
)

train_dataset = datasets.CIFAR10(
    root="../datasets", train=True, transform=train_transform, download=True
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
)

test_dataset = datasets.CIFAR10(
    root="../datasets", train=False, transform=test_transform, download=True,
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8
)
# 定义残差块 较低深度的ResNet使用BasicBlock
class BasicBlock(nn.Module):
    expansion = 1  # 每个残差块输出通道数与输入通道数的扩展倍数

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
#
# 2.定义ResNet-18模型


class ResNet18(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], stride=1
        )  # 3x32x32 → 64x32x32
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], stride=2
        )  # 64x32x32 → 128x16x16
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], stride=2
        )  # 128x16x16 → 256x8x8
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], stride=2
        )  # 256x8x8 → 512x4x4
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 3×32×32 → 64×32×32
        x = self.layer1(x)  # 64×32×32 → 64×32×32
        x = self.layer2(x)  # 64×32×32 → 128×16×16
        x = self.layer3(x)  # 128×16×16 → 256×8×8
        x = self.layer4(x)  # 256×8×8 → 512×4×4
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


model = ResNet18(BasicBlock, [2, 2, 2, 2], NUM_CLASSES).to(
    DEVICE
)  # 每层的残差块数量（2+2+2+2=8个块 × 2层/块 = 16个卷积层 + 1个初始卷积 = 17层）

# 3.定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)

# optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=WEIGHT_DECAY)

# 学习率调度器
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[100, 150, 175], gamma=0.1
)

# 余弦退火 可选
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        progress_bar(
            batch_idx,
            len(train_loader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (running_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    
    return train_loss, train_acc
    # 5.测试模型


def test(epoch):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total

    return test_loss, test_acc

    # 6.绘图


def plot(train_loss, train_acc, test_loss, test_acc):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(15, 5))

    # 损失对比图
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, "b-", label="Training Loss", alpha=0.7)
    plt.plot(epochs, test_loss, "r-", label="Test Loss", alpha=0.7)
    plt.title("Training vs Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 准确率对比图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, "b-", label="Training Accuracy", alpha=0.7)
    plt.plot(epochs, test_acc, "r-", label="Test Accuracy", alpha=0.7)
    plt.title("Training vs Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 训练模型
if __name__ == "__main__":
    os.makedirs("model_v", exist_ok=True)
    best_accuracy = 0.0

    # 初始化历史记录列表
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(EPOCHS):
        start = time.time()
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        scheduler.step()  # 更新学习率

        # 记录历史数据
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
            f"epoch_time: {time.time()}-{start}"
        )
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": test_acc,
                },
                "model_v/best_model.pth",
            )
            print(f"New best model saved with accuracy: {test_acc:.2f}%")

    print(f"Training completed! Best accuracy: {best_accuracy:.2f}%")

    plot(train_losses, train_accs, test_losses, test_accs)
