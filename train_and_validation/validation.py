#!/user/bin/python
# author jeff

import torch
from torch import nn
import torch.nn.functional as F

def validation(model, test_loader):
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0
    device='cpu'

    with torch.no_grad():
        for id,(data, target) in enumerate(test_loader):
            # if id==0:
            #     print("测试集：",data[0]) #这边同样DPSGD的验证集也是浮点型的
            data, target = data.to(device), target.to(device)
            output = model(data.to(torch.float32))
            test_loss += F.cross_entropy(output, target.to(torch.long), reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_examples += len(data)

    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{num_examples} ({test_acc:.2f}%)')

    return test_loss, test_acc

def validation_geo(model, test_loader):
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    with torch.no_grad():
        for id, data in enumerate(test_loader):
            # if id==0:
            #     print("测试集：",data[0]) #这边同样DPSGD的验证集也是浮点型的
            # data, target = data.to(device), target.to(device)
            # output = model(data.to(torch.float32))
            data = data.to(device)
            pred = model(data).argmax(dim=1)  # 取概率最高的结果的index作为pred的class
            correct += (pred[data.test_mask] == data.y[data.test_mask]).sum()  # 正确分类的数量
            # correct = (pred == data.y).sum()
            # acc = int(correct) / int(data.test_mask.sum())  # 正确分类数/总个数
        acc = int(correct) / int(data.test_mask.sum())  # 正确分类数/总个数

    return acc