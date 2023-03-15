import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader

from data.util.custom_tensor_dataset import CustomTensorDataset
import matplotlib.pyplot as plt


def test1():
    eps_list=[]
    p_list=[]
    for i in range(100):
        p=i*0.01
        eps=np.log(p/(1-p))
        p_list.append(p)
        eps_list.append(eps)
    plt.plot(p_list, eps_list)
    plt.xlabel('p')
    plt.ylabel('eps')
    plt.title('W-RR')
    plt.legend(['W-RR'], loc='best')
    plt.show()
    # c=list(zip(torch.split(a,2),torch.split(b,2)))
    # c=list(zip(a,b))
    # print(c)
    # transform = torchvision.transforms.ToTensor()
    # for e in c:
    #     print(e)
    #     print("-------")
    #     for j in e:
    #         print(j)
    #         print(j.size(0))
    #     p=CustomTensorDataset(e,transform)

    #现在的问题演变成T可以，但是C不可以，问题在于这个list zip，我们要进一步看下

if __name__=="__main__":
    test1()