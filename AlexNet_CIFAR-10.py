import torch
import torchvision
import torchvision.transforms as transforms
import wandb as wandb
from torch import nn
from torch.utils import data
from tqdm import tqdm


trainset = torchvision.datasets.CIFAR10(root = './FashionMNIST', train = True,
                                           download = True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=0, pin_memory=True) # 打乱，包装成batchsize

testset = torchvision.datasets.CIFAR10(root = './FashionMNIST',
                                          train = False, download = True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=0, pin_memory=True)
device = torch.device("cuda:0") # 选择cpu或者GPU
#device=torch.device("cpu")

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 6, 4),  # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2),  # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 2),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 2),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

model = AlexNet()
model.to(device) # 选择cpu或者gpu
model.load_state_dict(torch.load('AlexNet.pkl'))
criterion = nn.CrossEntropyLoss() # 交叉熵损失
optimizer = torch.optim.SGD([{'params': model.parameters()}],
                                lr=0.005, weight_decay=5e-4, momentum=0.9) # 随机梯度优化策略


proc_bar=tqdm(total=100)
for epoch in range(10):
        # 训练
        model.train()
        #for i, (img, label) in tqdm(enumerate(trainloader)):
        # 测试
        accuracy = 0
        model.eval()
        testlen = 0

        #for img, label in tqdm(trainloader,dynamic_ncols=True):
        for img, label in (trainloader):
            img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True).long() # 加载数据
            #print(img.size())
            output = model(img) # 计算结果
            loss = criterion(output, label) # 计算损失
            optimizer.zero_grad()
            loss.backward() # 反向传播
            optimizer.step() # 优化器更新


        for i, (img, label) in enumerate(testloader):
            img, label = img.to(device), label.to(device).long()
            output = model(img)
            output = output.max(dim=1)[1]  # 预测是哪一类
            accuracy += (output == label).sum().item()  # 准确率计算
            testlen += len(output)
        accuracy = accuracy / testlen  # 准确率计算
        proc_bar.update(100 / 80)
        proc_bar.set_description(f"epoch={epoch:02d},accuracy={accuracy:02f}")
        # wandb.log({'accuracy': accuracy}) # 可视化
proc_bar.close()

torch.save(model.state_dict(), 'AlexNet.pkl')#保存模型
print("save successfully")