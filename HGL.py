import torch
import torch.nn as nn
from torch.distributions import Normal


class HGL(nn.Module):
    """
    HGL: High-dimensional Gaussian distribution layers
    主要用于无监督聚类，假设分布存在"高内聚，低耦合"的特性，
    无监督意味着没有训练标签，HGL假定所有的样本出现概率为4
    调节联合高斯概率密度函数的参数使得所有样本的概率在4附近的过程中
    会形成高斯竞争，通过最大池化可以获得更新权限
    从而实现聚类
    """
    def __init__(self, dim=4, units=5, sef=3, init_mode='normal'):
        super(HGL, self).__init__()
        """
        dim: 输入特征维度
        units: 高斯单元数
        sef: Spatial extension factor(空间延展系数)
        init_mode: 参数初始化模式
        """
        # 初始化高维高斯层的参数分布，需要保证一定的空间延展能力，以便接下来形成高斯竞争
        if init_mode in ['normal', 0, '0']:
            self.mean = nn.Parameter(torch.zeros(dim*units))
        elif init_mode in ['randn', 1, '1']:
            self.mean = nn.Parameter(torch.randn(dim*units))
        self.sigma = nn.Parameter(torch.ones(dim*units)*sef)
        self.dim = dim
        self.units = units
        self.sef = sef

        # 形成高斯竞争的关键，锁定分布后会释放权限
        self.pool = nn.MaxPool1d(units, 1)

    def forward(self, din):
        # 生成高斯概率密度函数
        # normal = Normal(self.mean, torch.abs(self.sigma))
        # 防止出现nan影响训练
        normal = Normal(self.mean, torch.where(torch.isnan(self.sigma), torch.full_like(self.sigma, 0.001), torch.abs(self.sigma)))
        # 计算单个维度的高斯分布概率
        din = din.repeat(1, self.units)
        din = normal.log_prob(din).exp()
        # 计算联合概率密度
        din = din.reshape(din.shape[0], self.units, int(din.shape[1]/self.units))
        din = torch.prod(din, dim=-1)
        # 返回竞争胜利者和对应的下标
        return self.pool(din), torch.argmax(din, dim=-1)



if __name__ == '__main__':
    # CPU测试
    x = torch.randn(16, 4)
    net = HGL(dim=4, units=5)
    print('CPU测试通过，模块输出维度为：', net(x)[0].shape)
    # GPU测试
    x = torch.randn(16, 4).cuda()
    net = HGL(dim=4, units=5).cuda()
    print('GPU测试通过，模块输出维度为：', net(x)[0].shape)
