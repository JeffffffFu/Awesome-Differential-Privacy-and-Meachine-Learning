import numpy as np
import torch
import torchvision
from sympy.physics.quantum.identitysearch import scipy
from torch.utils.data import TensorDataset, DataLoader

from data.util.custom_tensor_dataset import CustomTensorDataset
import matplotlib.pyplot as plt


def test1():
    import numpy as np

    def cartesian_to_polar(x):
        r = np.linalg.norm(x)
        theta = np.arccos(x[0] / r)
        phi = [1. for i in range(len(x)-1)]
        for i in range(len(phi)):
            phi[i] = np.arctan2(x[i + 1], x[0]) #以X[0]为标准计算反正切值
        return np.concatenate(([r, theta], phi))

    def polar_to_cartesian(p):
        r = p[0]
        theta = p[1]
        phi = p[2:]
        x=[1. for i in range(len(phi)+1)]
        x[0] = r * np.cos(theta)  #用这个求回X[0]没有问题
        for i in range(len(phi)):
            x[i + 1] = x[0] * np.tan(phi[i])
        return x

    # 示例
    v = np.array([3, 4, 5])
    p = cartesian_to_polar(v)
    print(p)  # [7.81, 0.93, -0.87, 0.18, -1.33]
    x = polar_to_cartesian(p)
    print(x)  # [3. 4. 5.]

def test2():
    deltaE=0.001
    C=0.001
    sigma=1.0
    listE=[]
    threshold=-2*C
    for i in range(1000):
        listE.append(2 * C * sigma * np.random.normal(0, 1) + deltaE)

    result = len(list(filter(lambda x: x < threshold, listE)))
    print(result)


def test3():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import hypergeom

    # 定义超几何分布的参数
    N = 9  # 总体的大小
    K = 5  # 总体中具有某种属性的物件数量
    n = 8 # 进行抽样的样本数量

    # 创建超几何分布对象
    hypergeom_dist = hypergeom(N, K, n)

    # 生成可能的成功事件数量范围
    x = np.arange(0, n + 1)

    # 计算每个成功事件数量的概率
    pmf = hypergeom_dist.pmf(x)

    # 绘制超几何分布的概率质量函数图像
    plt.plot(x, pmf, marker='o', linestyle='-')
    plt.xlabel('x')
    plt.ylabel('p')
    plt.title('Hypergeom')
    plt.show()



if __name__=="__main__":
    test3()