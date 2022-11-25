#!/user/bin/python
# author jeff
import gzip
from pathlib import Path
import pickle
import numpy
import os
import numpy as np

# 数据集存储压缩，这里数据集提前下载好了
# import torchvision
# from torchvision import datasets


def dataload_mnist_50000():
    # 编写存放路径
    DATA_PATH = Path("../data/dataset")
    PATH = DATA_PATH / "Mnist"
    FILENAME = "mnist.pkl.gz"
    # 创建文件夹
    PATH.mkdir(parents=True, exist_ok=True)

    #这个Mnist数据集自动会解压出50000个训练集，10000个验证集和10000个测试集
    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:  # 解压分别获取文件
        ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")

    # 如果CNN的model需要加下面这段reshape
    x_train = x_train.reshape(50000, 1, 28, 28)
    x_valid = x_valid.reshape(10000, 1, 28, 28)
    x_test = x_test.reshape(10000, 1, 28, 28)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

#将验证集合并到训练集
def dataload_mnist_60000():
    # 编写存放路径
    DATA_PATH = Path("../data/dataset")
    PATH = DATA_PATH / "Mnist"
    FILENAME = "mnist.pkl.gz"
    # 创建文件夹
    PATH.mkdir(parents=True, exist_ok=True)

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:  # 解压分别获取文件
        ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")

    # 因为x_valid，y_valid没用到，我们把它合并到x_train, y_train中,这里train里面的样本数从50000到60000
    x_train = numpy.concatenate((x_train, x_valid), 0)
    y_train = numpy.concatenate((y_train, y_valid), 0)

    # 如果CNN的model需要加下面这段reshape
    # 如果是Net2nn则不需要
    x_train = x_train.reshape(60000, 1, 28, 28)
    x_valid = x_valid.reshape(10000, 1, 28, 28)
    x_test = x_test.reshape(10000, 1, 28, 28)

    return x_train, y_train, x_valid, y_valid, x_test, y_test


# 只取5500个样本训练数据
def dataload_mnist_5500():
    # 编写存放路径
    DATA_PATH = Path("../data/dataset")
    PATH = DATA_PATH / "Mnist"
    FILENAME = "mnist.pkl.gz"
    # 创建文件夹
    PATH.mkdir(parents=True, exist_ok=True)

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:  # 解压分别获取文件
        ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")

    # 因为x_valid，y_valid没用到，我们把它合并到x_train, y_train中,这里train里面的样本数从50000到60000
    x_train = numpy.split(x_train, [5500])
    y_train = numpy.split(y_train, [5500])
    x_train = x_train[0]
    y_train = y_train[0]
    print("x_train_shape:", x_train.shape)
    print("y_train_shape:", y_train.shape)

    x_test = numpy.split(x_test, [5500])
    y_test = numpy.split(y_test, [5500])
    x_test = x_test[0]
    y_test = y_test[0]
    print("x_test_shape:", x_test.shape)
    print("y_test_shape:", y_test.shape)

    # 如果CNN的model需要加下面这段reshape
    x_train = x_train.reshape(5500, 1, 28, 28)
    x_valid = x_valid.reshape(10000, 1, 28, 28)
    x_test = x_test.reshape(5500, 1, 28, 28)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

if __name__=="__main__":
    dataload_mnist_5500()
