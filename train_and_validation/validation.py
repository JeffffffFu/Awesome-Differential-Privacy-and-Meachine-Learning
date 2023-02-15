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