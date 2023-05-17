#!/user/bin/python
# author jeff
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import DataLoader as GeoDataLoader
import numpy as np


# 该采样每次采样的数量不等，即泊松采样，minibatch_size是采样多次后的平均采样数
class IIDBatchSampler:
    def __init__(self, dataset, minibatch_size, iterations):
        self.length = len(dataset)  # 总的样本个数
        self.minibatch_size = minibatch_size  # batch大小
        self.iterations = iterations  # batch个数

    def __iter__(self):
        for _ in range(self.iterations):  # 共iterations次

            # torch.rand：返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数，这里返回length个即总样本个数的随机数
            # self.minibatch_size / self.length=256/4500
            # np.where:输出满足条件 (即非0) 元素的坐标 (等价于numpy.nonzero)。这里的坐标以tuple的形式给出
            # 也就是说这里每轮会有不固定的坐标数，为什么每轮要不固定而不能固定呢？到底采样的意思是什么呢？这里的意思就是每轮的采样数量不等，但是最后的采样数量平均数等于给定的值
            indices = np.where(torch.rand(self.length) < (self.minibatch_size / self.length))[0]  # 这里具体得到坐标的数值
            # print("-------------------")
            # print(indices)
            if indices.size > 0:
                yield indices

    def __len__(self):
        return self.iterations


# 该采样每次抽取的数量相等,不放回
# 这里的不放回我们认为是每轮采样每个样本不放回，但是下一轮又重新开始。也就是说单轮不会有样本重复，但是下一轮可能会有样本重复
class EquallySizedAndIndependentBatchSamplerWithoutReplace:
    def __init__(self, dataset, minibatch_size, iterations):
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        for _ in range(self.iterations):
            # numpy.random.choice(a, size=None, replace=True, p=None)
            # 从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
            # replace:True表示可以取相同数字（有放回），False表示不可以取相同数字（无放回）
            # 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同
            yield np.random.choice(self.length, self.minibatch_size, replace=False)  # 这里最后产生的是一组坐标

    def __len__(self):
        return self.iterations


# 该采样每次抽取的数量相等,放回
class EquallySizedAndIndependentBatchSamplerWithReplace:
    def __init__(self, dataset, minibatch_size, iterations):
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        for _ in range(self.iterations):
            # numpy.random.choice(a, size=None, replace=True, p=None)
            # 从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
            # replace:True表示可以取相同数字（有放回），False表示不可以取相同数字（无放回）
            # 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同
            yield np.random.choice(self.length, self.minibatch_size, replace=True)  # 这里最后产生的是一组坐标

    def __len__(self):
        return self.iterations


# 三种采样都有的文章《Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences》

# 均匀采样不放回（《Subsamlping Gaussian RDP》）
# 注意：SGM高斯下采样机制用的都是无放回的策略，在文章《Renyi Differential Privacy of the Sampled Gaussian Mechanism》中定义3有提到: without replacement
def get_data_loaders_uniform_without_replace(minibatch_size, microbatch_size, iterations, drop_last=True,
                                             use_gnn=False):
    # 这个函数才是主要的采样
    def minibatch_loader(dataset):  # 具体调用这个函数的时候会给对应入参
        if not use_gnn:
            return DataLoader(
                dataset,  # 给定原本数据
                batch_sampler=EquallySizedAndIndependentBatchSamplerWithoutReplace(dataset, minibatch_size, iterations)
                # DataLoader中自定义从数据集中取样本的策略
            )
        else:
            return GeoDataLoader(dataset)

    # 最里面一层的data，对上面组装好的minibatch进行操作，主要目的是将每个minibatch细分成多个microbatch
    # 这个不是关键，主要的是上面的采样
    def microbatch_loader(minibatch):
        if not use_gnn:
            return DataLoader(
                minibatch,
                batch_size=microbatch_size,  # 要进一步细分的数量
                # Using less data than allowed will yield no worse of a privacy guarantee,
                # and sometimes processing uneven batches can cause issues during training, e.g. when
                # using BatchNorm (although BatchNorm in particular should be analyzed seperately for privacy, since it's maintaining internal information about forward passes  over time without noise addition.)
                drop_last=drop_last,
                # 这个是对最后的未完成的batch来说的，如果设置为True：比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…
            )
        else:
            return GeoDataLoader(
                minibatch,
                batch_size=microbatch_size,  # 要进一步细分的数量
                # Using less data than allowed will yield no worse of a privacy guarantee,
                # and sometimes processing uneven batches can cause issues during training, e.g. when
                # using BatchNorm (although BatchNorm in particular should be analyzed seperately for privacy, since it's maintaining internal information about forward passes  over time without noise addition.)
                drop_last=drop_last,
                # 这个是对最后的未完成的batch来说的，如果设置为True：比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…
            )

    return minibatch_loader, microbatch_loader  # 返回的是函数，不是数据，这里要注意了


# 均匀采样放回
def get_data_loaders_uniform_with_replace(minibatch_size, microbatch_size, iterations, drop_last=True):
    # 这个函数才是主要的采样
    def minibatch_loader(dataset):  # 具体调用这个函数的时候会给对应入参
        return DataLoader(
            dataset,  # 给定原本数据
            batch_sampler=EquallySizedAndIndependentBatchSamplerWithReplace(dataset, minibatch_size, iterations)
            # 自定义从数据集中取样本的策略，这边用IIDBatchSample函数进行样本抽取
        )

    # 最里面一层的data，对上面组装好的minibatch进行操作，主要目的是将每个minibatch细分成多个microbatch
    # 这个不是关键，主要的是上面的采样
    def microbatch_loader(minibatch):
        return DataLoader(
            minibatch,
            batch_size=microbatch_size,  # 要进一步细分的数量
            # Using less data than allowed will yield no worse of a privacy guarantee,
            # and sometimes processing uneven batches can cause issues during training, e.g. when
            # using BatchNorm (although BatchNorm in particular should be analyzed seperately for privacy, since it's maintaining internal information about forward passes  over time without noise addition.)
            drop_last=drop_last,
            # 这个是对最后的未完成的batch来说的，如果设置为True：比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…
        )

    return minibatch_loader, microbatch_loader  # 返回的是函数，不是数据，这里要注意了


# 泊松采样
def get_data_loaders_possion(minibatch_size, microbatch_size, iterations, drop_last=True):
    # 这个函数才是主要的采样
    def minibatch_loader(dataset):  # 具体调用这个函数的时候会给对应入参
        return DataLoader(
            dataset,  # 给定原本数据
            batch_sampler=IIDBatchSampler(dataset, minibatch_size, iterations)
            # 自定义从数据集中取样本的策略，这边用IIDBatchSample函数进行样本抽取
        )

    # 最里面一层的data，对上面组装好的minibatch进行操作，主要目的是将每个minibatch细分成多个microbatch
    # 这个不是关键，主要的是上面的采样
    def microbatch_loader(minibatch):
        return DataLoader(
            minibatch,
            batch_size=microbatch_size,  # 要进一步细分的数量
            # Using less data than allowed will yield no worse of a privacy guarantee,
            # and sometimes processing uneven batches can cause issues during training, e.g. when
            # using BatchNorm (although BatchNorm in particular should be analyzed seperately for privacy, since it's maintaining internal information about forward passes  over time without noise addition.)
            drop_last=drop_last,
            # 这个是对最后的未完成的batch来说的，如果设置为True：比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…
        )

    return minibatch_loader, microbatch_loader  # 返回的是函数，不是数据，这里要注意了


'''
#test,第一种和第二种其实一样，后面三个都有对应的隐私分析。默认现在的操作都是用第一种操作，然后给到第三种的隐私预算。
b=torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
print("data数据,数据总量20：",b)
dataset=TensorDataset(b)
print("1、基于传统DataLoader（当前SGD用的数据处理方法）,batchsize=5-----------") #也叫随机重组 在《Differentially Private Model Publishing for Deep Learning》
train_dl = DataLoader(dataset, batch_size=5, shuffle=True)
for data in train_dl:
    print(data)
print("2、基于uniform sampling(without replacement),batchsize=5-----------")
minibatch_loader, microbatch_loader = get_data_loaders_uniform_without_replace(5, 1, 4)
a=minibatch_loader(dataset)
for data in a:
    print(data)
print("3、基于uniform sampling(with replacement),batchsize=5-----------")
minibatch_loader, microbatch_loader = get_data_loaders_uniform_with_replace(5, 1, 4)
a=minibatch_loader(dataset)
for data in a:
    print(data)
print("4、基于possion sampling,batchsize=5-----------")
minibatch_loader, microbatch_loader = get_data_loaders_possion(5, 1, 4)
a=minibatch_loader(dataset)
for data in a:
    print(data)
'''
