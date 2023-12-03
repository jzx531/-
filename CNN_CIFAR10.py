import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils import data
from tqdm import tqdm,trange

augment_transform=transforms.Compose([transforms.Compose([transforms.RandomAffine
                                                          (degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10)]),
                                      transforms.RandomGrayscale(p=0.1)])
trainset = torchvision.datasets.CIFAR10(root = './FashionMNIST', train = True,
                                           download = True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=0, pin_memory=True) # 打乱，包装成batchsize

testset = torchvision.datasets.CIFAR10(root = './FashionMNIST',
                                          train = False, download = True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=0, pin_memory=True)
device = torch.device("cuda:0") # 选择cpu或者GPU

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 2 * 2, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 64 * 2 * 2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
        #return torch.softmax(x,dim=1)


model = CNN()
model.to(device) # 选择cpu或者gpu
model.load_state_dict(torch.load('CNN_CIFAR10.pkl'))
#print(model)
print("load successfully")
"""
for k,v in model.named_parameters():
     if k.find('conv'):
         v.requires_grad=False#固定卷积层参数
"""
for k,v in model.named_parameters():
     if k.find('fc'):
         v.requires_grad=False#固定连接层参数
#两种层最多固定一个
for name, param in model.named_parameters():
    if  param.requires_grad:
        print(name)
criterion = nn.CrossEntropyLoss() # 交叉熵损失
optimizer = torch.optim.SGD([{'params': model.parameters()}],
                                lr=0.005, weight_decay=5e-4, momentum=0.9) # 随机梯度优化策略

epoches=10
proc_bar=tqdm(total=100,position=1)
for epoch in range(epoches):
        # 训练
        model.train()
        #for i, (img, label) in tqdm(enumerate(trainloader)):
        # 测试
        accuracy = 0
        model.eval()
        testlen = 0


        #for img, label in tqdm(trainloader,dynamic_ncols=True):
        for i,(img, label) in enumerate(tqdm(trainloader,position=0)):
            img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True).long() # 加载数据
            """
            if random.random() < 0.2:
                t=img
                for i in range(img.size(0)):
                    t=transforms.ToPILImage()(t[i])
                    t= augment_transform(t)
                    t=torch.tensor(t).to(device, non_blocking=True)
                    t = t.unsqueeze(dim=0)
                    img[i]=t
            """
            output = model(img) # 计算结果
            loss = criterion(output, label) # 计算损失
            optimizer.zero_grad()
            loss.backward() # 反向传播
            optimizer.step() # 优化器更新

            #iters = epoch * len(trainloader) + i
            #if iters % 10 == 0:
             #   wandb.log({'loss': loss}) # 可视化
        for i, (img, label) in enumerate(testloader):
            img, label = img.to(device), label.to(device).long()
            output = model(img)
            output = output.max(dim=1)[1]  # 预测是哪一类
            accuracy += (output == label).sum().item()  # 准确率计算
            testlen += len(output)
        accuracy = accuracy / testlen  # 准确率计算
        proc_bar.update(100 / epoches)
        proc_bar.set_description(f"epoch={epoch+1:02d},accuracy={accuracy:02f}")
        # wandb.log({'accuracy': accuracy}) # 可视化
proc_bar.close()

torch.save(model.state_dict(), 'CNN_CIFAR10.pkl')#保存模型
print("save successfully")