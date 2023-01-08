import torch
from torch import nn
import torch.nn.functional as F


#opacus<=0.13.0，并行的训练，关键在virtual_step
def train_with_opacus(model, train_loader, optimizer,n_acc_steps):

    model.train()
    num_examples = 0
    correct = 0
    train_loss = 0

    print("train_loader:",len(train_loader))
    print("n_acc_steps:",n_acc_steps)

    rem = len(train_loader) % n_acc_steps  #这个什么意思
    num_batches = len(train_loader)
    print("num_batches:",num_batches)
    print("rem:",rem)  #为什么是2
    num_batches -= rem

    for batch_idx, (data, target) in enumerate(train_loader):  #235*256(batchsize)

        if batch_idx > num_batches - 1:
            break

        output = model(data)  # 这要是这里要做升维
        #loss = criterion(output, target.to(torch.long))  # 定义损失
        loss = F.cross_entropy(output, target)
        # output = model(data)
        # loss = criterion(output, target)
        loss.backward()

        #多少次做一次step
        if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
        else:
            with torch.no_grad():
                # accumulate per-example gradients but don't take a step yet
                # 梯度裁剪，求和但不下降不加噪。但是这里的梯度不是每个样本了吧，看起来是256个样本的平均梯度了
                optimizer.virtual_step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += F.cross_entropy(output, target, reduction='sum').item()
        num_examples += len(data)



    train_loss /= num_examples
    train_acc = 100. * correct / num_examples

    # print(f'Train set: Average loss: {train_loss:.4f}, '
    #         f'Accuracy: {correct}/{num_examples} ({train_acc:.2f}%)')


    return train_loss, train_acc



#opacus=-0.13.0
#MAX_PHYSICAL_BATCH_SIZE=2048  #去研究一下这个有什么作用
#from opacus.utils.batch_memory_manager import BatchMemoryManager

# def train_privacy_opacus2(model, train_loader, criterion, optimizer,delta,privacy_engine):
#     model.train()
#     train_loss = 0.0
#     correct = 0
#
#     with BatchMemoryManager(
#             data_loader=train_loader,
#             max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
#             optimizer=optimizer
#     ) as memory_safe_data_loader:
#
#         for data, target in memory_safe_data_loader:  # batch之前组装到data数据集里的,pytorch的MBDG统一用这种方式进行,会按序列一个个btach训练
#             optimizer.zero_grad()  # 梯度清空
#             output = model(data.to(torch.float32))  # 这要是这里要做升维
#             loss = criterion(output, target)  # 定义损失
#
#             loss.backward()  # 梯度求导
#
#             optimizer.step()  # 参数优化更新
#
#             train_loss += loss.item()  # 损失累加
#             prediction = output.argmax(dim=1, keepdim=True)  # 将one-hot输出转为单个标量
#             correct += prediction.eq(target.view_as(prediction)).sum().item()  # 比较得到准确率
#
#             # 这里调用隐私分析得到epsilon
#             # epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
#             # epsilon=PrivacyEngine.get_epsilon(delta=delta)
#         #epsilon = privacy_engine.get_epsilon(delta=delta)
#         epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(delta=delta)
#
#
#
#     return train_loss / len(train_loader), correct / len(train_loader.dataset), epsilon, best_alpha  # 返回平均损失和平均准确率