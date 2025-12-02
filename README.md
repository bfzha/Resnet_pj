## 数据集介绍：
https://blog.csdn.net/weixin_45954454/article/details/114519299

https://www.cnblogs.com/mengtao-wang/p/18888373

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

## 超参数、正则化和学习率调度
设置超参数

BATCH_SIZE = 128

NUM_CLASSES = 10 

LR = 0.1

MOMENTUM = 0.9

WEIGHT_DECAY = 5e-4

EPOCHS = 100      

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weight_decay (权重衰减)：

这是一种正则化技术，用于防止模型过拟合。它通过在损失函数中添加 L2 正则化项来实现。作用是惩罚大的权重值，使模型的权重更小，从而降低模型复杂度。它在整个训练过程中都是持续的，不会随时间改变

学习率调度器：

用于在训练过程中动态调整学习率。在代码中，MultiStepLR 会在第 100、150、200 轮将学习率乘以 0.1。目的是在训练后期使用较小的学习率，帮助模型更好地收敛

## 数据增强
### CIFAR-10 专用的标准化参数
MEAN_CIFAR10 = [0.4914, 0.4822, 0.4465]
STD_CIFAR10 = [0.2470, 0.2435, 0.2616]

1. RandomCrop(32, padding=4)：模拟图像小范围位移，提高平移不变性。

2. RandomHorizontalFlip(p=0.5)：左右翻转，适用于对象左右对称的类，增加多样性。

3. transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)：CIFAR-10类别少，模型更易过拟合到训练集颜色。更强的色彩扰动可迫使模型学习更鲁棒的特征，而非依赖特定颜色。

4. transforms.RandomRotation(degrees=15)：适度增加旋转角度可提升模型对物体方向变化的鲁棒性。
 
5. transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)) 仿射平移

6. transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')：此方法通过随机遮挡部分图像，能有效迫使模型关注全局特征而非局部纹理，对防止过拟合、提升泛化能力效果显著。


# 1.准备数据集
MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2470, 0.2435, 0.2616]

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), # 以0.5的概率水平翻转图像，增加数据多样性
    transforms.RandomCrop(32, padding=4), # 32x32的图像，填充4个像素后再裁剪回32x32 模拟图像小范围位移，提高平移不变性

    #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), 
    #transforms.RandomRotation(degrees=15), 
    #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 

    transforms.ToTensor(), 
    transforms.Normalize(MEAN, STD), 
    
    #transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random') 
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

train_dataset = datasets.CIFAR10(root='../datasets', train=True, transform=train_transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.CIFAR10(root='../datasets', train=False, transform=test_transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

## 任务

数据增强中1，2是基础，3~6需要做消融实验：要科学评估每种增强的效果，最好的方法是进行消融实验。设置一组只使用RandomCrop和RandomHorizontalFlip的基准实验，然后依次加入增强后的ColorJitter、RandomErasing等，观察每个改动对最终测试准确率的影响。 

### 实验设计

EXP A：消融实验
Exp 1: ResNet-18 SGD scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 80], gamma=0.1) lr = 0.1 
1. 基准实验：只使用RandomCrop和RandomHorizontalFlip
2. 消融实验1：加入ColorJitter
3. 消融实验2：加入RandomRotation
4. 消融实验3：加入RandomAffine
5. 消融实验4：加入RandomErasing

EXP B：根据哪个效果最好，在进行下面实验

Exp 1: ResNet-18(base) SGD scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 80], gamma=0.1) lr = 0.1

Exp 2: ResNet-18 SGD 余弦退火 lr = 0.1   

Exp 3: ResNet-18 ADAM 余弦退火 lr=3e-4 

Exp 4: ResNet-18 + SE模块 SGD 余弦退火 lr = 0.1

Exp 5: ResNet-34 SGD scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 80], gamma=0.1) lr = 0.1


## 结果分析与可视化
1.不同训练策略和数据增强方法对模型性能的影响（图+表）

2.参考：https://www.cnblogs.com/mengtao-wang/p/18888373

涵盖6.2 精确率，每一类的精确率，6.3 混淆矩阵与分类报告 6.4 ROC曲线图