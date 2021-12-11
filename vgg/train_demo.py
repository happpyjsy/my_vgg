import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# 自己制作数据集
class MnistDataset(Dataset):

    def __init__(self, image_path, image_label, transform=None):
        super(MnistDataset, self).__init__()
        self.image_path = image_path  # 初始化图像路径列表
        self.image_label = image_label  # 初始化图像标签列表
        self.transform = transform  # 初始化数据增强方法

    def __getitem__(self, index):
        """
        获取对应index的图像，并视情况进行数据增强
        """
        image = Image.open(self.image_path[index])
        # image = np.array(image)
        label = float(self.image_label[index])

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label)

    def __len__(self):
        return len(self.image_path)


def get_path_label(img_root, label_file_path):
    """
    获取数字图像的路径和标签并返回对应列表
    @para: img_root: 保存图像的根目录
    @para:label_file_path: 保存图像标签数据的文件路径 .csv 或 .txt 分隔符为','
    @return: 图像的路径列表和对应标签列表
    """
    data = pd.read_csv(label_file_path, names=['img', 'label'])
    data['img'] = data['img'].apply(lambda x: img_root + x)
    return data['img'].tolist(), data['label'].tolist()


# 获取训练集路径列表和标签列表
train_data_root = r'C:\Users\LENOVO\Desktop\手写数据集\dataset\dataset\MNIST\mnist_data/train/'
train_label = r'C:\Users\LENOVO\Desktop\手写数据集\dataset\dataset\MNIST\mnist_data/train.txt'
train_img_list, train_label_list = get_path_label(train_data_root, train_label)
# 训练集dataset
train_dataset = MnistDataset(train_img_list,
                             train_label_list,
                             transform=transforms.Compose([transforms.Resize((32, 32)),
                                                           transforms.ToTensor()]))

# 获取测试集路径列表和标签列表
test_data_root = r'C:\Users\LENOVO\Desktop\手写数据集\dataset\dataset\MNIST\mnist_data/test/'
test_label = r'C:\Users\LENOVO\Desktop\手写数据集\dataset\dataset\MNIST\mnist_data/test.txt'
test_img_list, test_label_list = get_path_label(test_data_root, test_label)
# 测试集sdataset
test_dataset = MnistDataset(test_img_list,
                            test_label_list,
                            transform=transforms.Compose([transforms.Resize((32, 32)),
                                                          transforms.ToTensor()]))

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True
)

class VGG(nn.Module):
    """
    VGG builder
    """

    def __init__(self, arch: object, num_classes=1000) -> object:
        super(VGG, self).__init__()
        self.in_channels = 1  # 因为手写体数据集是单通道灰度图  如果是正常彩色RGB图像 这里是3
        self.conv3_64 = self.__make_layer(64, arch[0])
        self.conv3_128 = self.__make_layer(128, arch[1])
        self.conv3_256 = self.__make_layer(256, arch[2])
        self.conv3_512a = self.__make_layer(512, arch[3])
        self.conv3_512b = self.__make_layer(512, arch[4])
        # self.fc1 = nn.Linear(7 * 7 * 512, 4096)
        self.fc1 = nn.Linear(1 * 1 * 512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        #nn.BatchNormm1d(num_features)num_features需要归一化的那一维的维度
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))  # same padding
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3_64(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_128(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_256(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512a(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512b(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        return self.fc3(out)


def VGG_11():
    return VGG([1, 1, 2, 2, 2], num_classes=10)


def VGG_13():
    return VGG([2, 2, 2, 2, 2], num_classes=10)


def VGG_16():
    return VGG([2, 2, 3, 3, 3], num_classes=10)


def VGG_19():
    return VGG([2, 2, 4, 4, 4], num_classes=10)

model = VGG_11()


MAX_EPOCH = 10
# BATCH_SIZE = 64
log_interval = 30 # 这个数字代表训练多少张图片打印一次信息
LR = 0.001


# ============================ step 2/5 模型 ============================
if torch.cuda.is_available():
#     model=nn.DataParallel(model)
    model.cuda()

# ============================ step 3/5 损失函数 ============================
criterion=nn.CrossEntropyLoss() #交叉熵损失函数
# ============================ step 4/5 优化器 ============================
optimizer=optim.Adam(model.parameters(),lr=LR, betas=(0.9, 0.99))
# 选择优化器---自适应优化器

model.train()
accurancy_global = 0.0
for epoch in range(MAX_EPOCH):
    loss_mean = 0.
    correct = 0.
    total = 0.
    running_loss = 0.0
    # accurancy_global = 0.0

    for i, data in enumerate(train_loader):
        img, label = data
        img = Variable(img)
        label = Variable(label)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()

        # 前向传播
        out = model(img)
        optimizer.zero_grad()
        # 归0梯度，即将梯度初始化为零（因为一个batch的loss关于weigh的导数是所有samplede的loss的weight的倒数的累加和
        loss = criterion(out, label.long())  # 得到损失函数

        print_loss = loss.data.item()

        loss.backward()  # 反向传播
        optimizer.step()  # 优化
        _, predicted = torch.max(out.data, 1)
        total += label.size(0)
        if(i+1)%log_interval==0:
          print("==================================")
          print("源数据标签: ",label)
          print("==================================")
          print("预测结果：",predicted)
          print("相等的结果为：",predicted==label)

        correct += (predicted == label).sum()
        if (i + 1) % log_interval == 0:
            print("准确率为：",correct.item() / total)
    #         print(correct.item())
    print("============================================")

    accurancy = correct.item() / total
    if accurancy > accurancy_global:
        torch.save(model.state_dict(), r'C:\Users\LENOVO\Desktop\手写数据集\dataset\dataset\MNIST\mnist_data\weights/'+str(accurancy)+'.pt')
        print("准确率由：", accurancy_global, "上升至：", accurancy, "已更新并保存权值为weights/best.pkl")
        accurancy_global = accurancy
    print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, 100 * accurancy))

    # -----------测试集
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for i, data in enumerate(test_loader):
        img, label = data
        img = Variable(img)
        label = Variable(label)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        out = model(img)
        loss = criterion(out, label.long())  # 得到损失函数
        eval_loss += loss.data.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))

print("训练完毕")



