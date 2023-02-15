import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader

from data.util.custom_tensor_dataset import CustomTensorDataset


def test1():
    a=([1,2,4,5],[3,5,5,6])
    b=([3,2],[6,2])
    a=([1,2],[3,5])
    b=([3],[6])
    a=torch.Tensor(a)
    b=torch.Tensor(b)
    transform = torchvision.transforms.ToTensor()

    print((a,b))
    c=CustomTensorDataset((a,b),transform)

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