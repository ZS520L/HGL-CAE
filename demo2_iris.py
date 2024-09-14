import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 定义 HGL 模型
class HGL(nn.Module):
    def __init__(self, dim=4, units=3, sef=1, init_mode='randn'):
        super(HGL, self).__init__()

        if init_mode in ['normal', 0, '0']:
            self.mean = nn.Parameter(torch.zeros(dim * units))
        elif init_mode in ['randn', 1, '1']:
            self.mean = nn.Parameter(torch.randn(dim * units))

        self.sigma = nn.Parameter(torch.ones(dim * units) * sef)
        self.dim = dim
        self.units = units
        self.sef = sef

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
        return torch.max(din, dim=1, keepdim=True)

# 加载鸢尾花数据集
iris = load_iris()
X_numpy = iris.data
labels_true = iris.target

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numpy)

# 将数据转换为 Tensor
X = torch.from_numpy(X_scaled.astype(np.float32))

# 设置 HGL 模型的参数
dim = X_numpy.shape[1]
units = 3  # 鸢尾花数据集有 3 个类别
sef = 1
init_mode = 'normal'

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
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# 获取聚类标签
_, cluster_labels = hgl(X)
cluster_labels = cluster_labels.squeeze().detach().numpy()

# 计算 NMI 指标
nmi = normalized_mutual_info_score(labels_true, cluster_labels)
print(f'Normalized Mutual Information (NMI): {nmi:.4f}')

# 可视化聚类结果
# 使用 PCA 将数据降维到 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_numpy)

plt.figure(figsize=(12, 5))

# 原始标签
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_true)
plt.title('原始数据标签')

# HGL 模型聚类结果
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels)
plt.title('HGL 模型聚类结果')

plt.show()
# 0.8642
