import random
import sys

import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import time


# 可视化数据
def use_svg_display():
    # 用矢量图显示
    set_matplotlib_formats('svg')


def set_figsize(figuresize=(5.5, 3.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figuresize


# 它每次返回 batch_size (批量大小) 个随机样本的特征和标签。
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


def linreg(X, w, b):
    return torch.mm(X, w) + b  # mm 作为矩阵乘法


def squared_loss(y_hat, y):
    return ((y_hat - y.view(y_hat.size())) ** 2) / 2


def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad.data / batch_size


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag',
                   'ankle  boot']
    return [text_labels[int(i)] for i in labels]


# 画出图像以及对应的标签函数
def show_fashion_mnist(images, labels):
    use_svg_display()
    _, figs = plt.subplots(nrows=1, ncols=len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view(28, 28).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    # 这里使用到了广播机制
    return (X_exp / partition)


# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进。
def evaluate_accuracy(data_iter, net, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    with torch.no_grad():  # 计算正确率的时候，关闭梯度
        for X, y in iter(data_iter):
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式，关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:
                if 'is_training' in net.__code__.co_varnames:  # 如果有is_training这个参数
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        return acc_sum / n


num_epochs, lr = 5, 0.1


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    # 每个参数的梯度都清零
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)  # 默认sgd
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
            epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


def load_data_fashion_mnist(batch_size):
    """
    加载fashion数据集
    :param batch_size: 加载的大小
    :return: train_iter, test_iter
    """
    mnist_train = torchvision.datasets.FashionMNIST(root='Datasets/FashionMNIST', train=True, download=True,
                                                    transform=transforms.ToTensor())  # 把 PIL 数据类型改成 Tensor
    mnist_test = torchvision.datasets.FashionMNIST(root="Datasets/FashionMNIST", train=False, download=True,
                                                   transform=transforms.ToTensor())
    if sys.platform.startswith('win'):
        num_workers = 0  # 0 代表不需要额外的进程来加速数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


# 对 x 形状转换
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.size(0), -1)


# 作图
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)


def corr2d(X, K):
    # 卷积
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))  # 用于生成一个从标准正态分布（均值为0，方差为1）中抽取的随机数张量。

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,  time %.1f sec' % (
            epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def load_data_fashion_mnist_in_ch5(batch_size, resize=None, root='Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into  memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_iter, test_iter


# 定义全局平均池化层
class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层通过将池化窗口设置为输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        # kernel_size 设置为输入的高和宽
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(Y))
        if self.conv3:
            X = self.conv3(X)
        return Y + X