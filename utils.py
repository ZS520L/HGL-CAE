import random
import torch
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def display(confusion, dataset='mnist', cluster_num=30):
    plt.figure(figsize=(25, 25))  # 设置图片大小

    # 1.热度图，后面是指定的颜色块，cmap可设置其他的不同颜色
    plt.imshow(confusion, cmap=plt.cm.Blues)
    plt.colorbar()  # 右边的colorbar

    # 2.设置坐标轴显示列表
    xindices = range(cluster_num)
    if dataset == 'mnist':
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        indices = range(10)
    elif dataset == 'fashion_mnist':
        classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        indices = range(10)
    xclasses = [str(i) for i in range(cluster_num)]
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    plt.xticks(xindices, xclasses, rotation=45)  # 设置横坐标方向，rotation=45为45度倾斜
    plt.yticks(indices, classes)

    # 3.设置全局字体
    # 在本例中，坐标轴刻度和图例均用新罗马字体['TimesNewRoman']来表示
    # ['SimSun']宋体；['SimHei']黑体，有很多自己都可以设置
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 4.设置坐标轴标题、字体
    plt.ylabel('True label')
    plt.xlabel('Cluster label')
    plt.title('HGL_CAE Confusion matrix on MNIST')

    # plt.xlabel('聚类标签')
    # plt.ylabel('真实标签')
    # plt.title('混淆矩阵', fontsize=12, fontfamily="SimHei")  # 可设置标题大小、字体

    # 5.显示数据
    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = confusion.max() / 2.

    for i in range(len(confusion)):  # 第几行
        for j in range(len(confusion[i])):  # 第几列
            plt.text(j, i, format(int(confusion[i][j]), fmt),
                     fontsize=16,  # 矩阵字体大小
                     horizontalalignment="center",  # 水平居中。
                     verticalalignment="center",  # 垂直居中。
                     color="white" if confusion[i, j] > thresh else "black")
    plt.show()


def TSNE_SHOW(model, dataloader, device='cuda:0', encoder=True):
    # 进行测试, 生成测试数据
    for i, (datas, labels) in enumerate(dataloader):
        batch_size = datas.size(0)
        datas = datas.to(device)
        if encoder:
            x = model(datas)
        else:
            x = datas
        x = x.cpu().detach().numpy().reshape(batch_size, -1)
        if i == 0:
            featureList = x
            labelsList = labels
        else:
            featureList = np.append(featureList, x, axis=0)
            labelsList = np.append(labelsList, labels, axis=0)
    # 降到2维
    x_encode = TSNE(n_components=2).fit_transform(featureList)  # 接着使用tSNE进行降维
    print(x_encode.shape)
    # 进行可视化
    cmap = plt.get_cmap('plasma', 10)  # 数字与颜色的转换

    # 获得可视化数据
    v_x = x_encode
    v_y = labelsList

    # 进行可视化
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1)

    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    labels1 = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    for key in classes:
        ix = np.where(v_y == key)
        ax.scatter(v_x[ix][:, 0], v_x[ix][:, 1], color=cmap(key), label=labels1[key])

    ax.legend()
    plt.show()


def Clustering_integration(Confusion_matrix):
    Confusion_matrix = torch.Tensor(Confusion_matrix)
    result = torch.zeros(Confusion_matrix.size(0), Confusion_matrix.size(0))
    for i, index in zip(range(Confusion_matrix.size(-1)), torch.max(Confusion_matrix, dim=0)[1].numpy()):
        index = index.item()
        result[:, index] = result[:, index] + Confusion_matrix[:, i]
    return result


def Confusion_matrix_show(confusion):
    plt.figure(figsize=(15, 15))  # 设置图片大小
    # 1.热度图，后面是指定的颜色块，cmap可设置其他的不同颜色
    plt.imshow(confusion, cmap=plt.cm.Blues)
    plt.colorbar()  # 右边的colorbar

    # 2.设置坐标轴显示列表
    indices = range(10)
    xindices = range(10)
    # classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    xclasses = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # xclasses = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    plt.xticks(xindices, xclasses, rotation=45)  # 设置横坐标方向，rotation=45为45度倾斜
    plt.yticks(indices, classes)

    # 3.设置全局字体
    # 在本例中，坐标轴刻度和图例均用新罗马字体['TimesNewRoman']来表示
    # ['SimSun']宋体；['SimHei']黑体，有很多自己都可以设置
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 4.设置坐标轴标题、字体
    plt.ylabel('True label')
    plt.xlabel('Cluster label')
    plt.title('HGL_CAE Confusion matrix on FashionMNIST')

    # plt.xlabel('聚类标签')
    # plt.ylabel('真实标签')
    # plt.title('混淆矩阵', fontsize=12, fontfamily="SimHei")  # 可设置标题大小、字体

    # 5.显示数据
    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = confusion.max() / 2.

    for i in range(len(confusion)):  # 第几行
        for j in range(len(confusion[i])):  # 第几列
            plt.text(j, i, format(int(confusion[i][j]), fmt),
                     fontsize=16,  # 矩阵字体大小
                     horizontalalignment="center",  # 水平居中。
                     verticalalignment="center",  # 垂直居中。
                     color="white" if confusion[i, j] > thresh else "black")
    # plt.savefig('test.jpg')
    # plt.savefig('test.png', bbox_inches='tight', dpi=600, pad_inches=0.0)

    plt.show()
    # plt.savefig('test.jpg')

# Confusion_matrix_show(Clustering_integration(res))
