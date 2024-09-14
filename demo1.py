import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 定义 HGL 模型
class HGL(nn.Module):
    def __init__(self, dim=4, units=5, sef=3, init_mode='normal'):
        super(HGL, self).__init__()

        if init_mode in ['normal', 0, '0']:
            self.mean = nn.Parameter(torch.zeros(dim * units))
        elif init_mode in ['randn', 1, '1']:
            self.mean = nn.Parameter(torch.randn(dim * units))

        self.sigma = nn.Parameter(torch.ones(dim * units) * sef)
        self.dim = dim
        self.units = units
        self.sef = sef
        self.pool = nn.MaxPool1d(units, 1)

    def forward(self, din):
        sigma = torch.abs(self.sigma)

        # 生成高斯分布
        normal = Normal(self.mean, sigma)

        # 扩展输入数据以匹配 units 数量
        din = din.repeat(1, self.units)

        # 计算对数概率并取指数，得到概率密度
        din = normal.log_prob(din).exp()

        # 重塑并计算联合概率密度
        din = din.reshape(din.shape[0], self.units, self.dim)
        din = torch.prod(din, dim=-1)

        # 返回池化后的结果和对应的索引（聚类标签）
        return self.pool(din), torch.argmax(din, dim=-1)

# 生成示例数据集
n_samples = 300
n_features = 2
centers = 4
X_numpy, labels_true = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features, random_state=42)

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numpy)

# 将数据转换为 Tensor
X = torch.from_numpy(X_scaled.astype(np.float32))

# 设置 HGL 模型的参数
dim = n_features
units = 4  # 设定为预期的簇数
sef = 1
init_mode = 'randn'

# 初始化 HGL 模型
hgl = HGL(dim=dim, units=units, sef=sef, init_mode=init_mode)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(hgl.parameters(), lr=0.01)
n_epochs = 1000

for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    # 前向传播
    prob, _ = hgl(X)
    
    # 计算负对数似然损失
    log_prob = torch.log(prob + 1e-8)
    loss = -torch.mean(log_prob)
    
    # 反向传播和优化
    loss.backward()
    optimizer.step()
    
    # 打印损失（可选）
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# 获取聚类标签
_, cluster_labels = hgl(X)
cluster_labels = cluster_labels.detach().numpy()

# 可视化聚类结果
plt.figure(figsize=(12, 5))

# 原始标签
plt.subplot(1, 2, 1)
plt.scatter(X_numpy[:, 0], X_numpy[:, 1], c=labels_true)
plt.title('原始数据标签')

# HGL 模型聚类结果
plt.subplot(1, 2, 2)
plt.scatter(X_numpy[:, 0], X_numpy[:, 1], c=cluster_labels)
plt.title('HGL 模型聚类结果')

plt.show()
