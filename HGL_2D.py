import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.distributions import Normal
from HGL import HGL

if __name__ == '__main__':
    # 测试并构建网络
    din = torch.randn(16, 2).cuda()
    g_units = 2
    net = HGL(2, g_units).cuda()
    print(net(din)[0].shape)
    # 准备数据集
    x = []
    y = []
    for i in range(400, 450):
        x.append([i/1000, i/500])
        y.append(4)
    for i in range(-300, -270):
        x.append([i/1000, i/500])
        y.append(4)
    # for i in range(100, 130):
    #     x.append(i / 1000)
    #     y.append(4)
    # for i in range(-500, -450):
    #     x.append(i / 1000)
    #     y.append(4)
    # for i in range(-130, -100):
    #     x.append(i / 1000)
    #     y.append(4)
    # for i in range(-1100, -1000):
    #     x.append(i / 1000)
    #     y.append(4)

    train_ids = torch.utils.data.TensorDataset(torch.Tensor(x), torch.Tensor(y))

    # DataLoader进行数据封装
    print('=' * 80)
    train_loader = torch.utils.data.DataLoader(dataset=train_ids, batch_size=8, shuffle=True)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
    bar = tqdm(range(150))
    for t in bar:
        for step, (x1, y1) in enumerate(train_loader):
            # Forward pass: Compute predicted y by passing x to the model
            # print(x1.shape)
            y_pred, _ = net(x1.cuda())

            # Compute and print loss
            loss = loss_func(y_pred.squeeze(1), y1.cuda())  # 计算损失函数

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
            loss.backward()  # loss反向传播
            optimizer.step()  # 反向传播后参数更新
        # print(loss)  # 可以查看历史
        bar.set_description(f"loss: {loss}")  # 看不到历史
    for params in enumerate(net.named_parameters()):
        print(params)
    #     for params in enumerate(net.named_parameters()):
    #         if 'mean' in params[1][0]:
    #             m = params[1][1].detach().cpu()
    #         if 'sigma' in params[1][0]:
    #             s = params[1][1].detach().cpu()
    #     x_linspace = torch.linspace(-2, 2, 1000)
    #     plt.clf()
    #     plt.scatter(torch.tanh(torch.Tensor(x)*net.sef)*net.sef, y, c='b')
    #     for i in range(g_units):
    #         plt.plot(x_linspace, Normal(m[i], s[i]).log_prob(x_linspace).exp())
    #     plt.title(f'epoch:{t}')
    #     plt.draw()
    #     plt.pause(0.05)
    # plt.ioff()
    # plt.show()

