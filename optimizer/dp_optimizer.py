#!/user/bin/python
# author jeff

import numpy as np
import torch
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.normal import Normal
from torch.optim import SGD, Adam, Adagrad, RMSprop

#cls表示你用的是哪个梯度下降的原函数，只有下面四种选择。
# *agrs,##kwargs为了解决不同梯度下降函数的参数问题，因为大家要的参数不同，统一用这两个进行入参填补，具体要传入的参数可以在创建这个DPOptimizer的时候输入
#比如SGD就不需要动能参数。

def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size, *args, **kwargs):
            # args表示剩余参数的值，kwargs在args之后表示成对键值对。

            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size  #batch_size

            for id,group in enumerate(self.param_groups):
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]
                # print("accum_grad:",  group['accum_grads'])
                # print("id:",len(group['accum_grads']))
        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad() #这个清零应该是和下面的不一样，这个是模型参数清零

        # 'param':{}, 'lr': 0.15, 'momentum': 0, 'dampening': 0, 'weight_decay': 0.01, 'nesterov': False, 'accum_grads':之前的梯度模型{}
        # accum_grads到底是什么,可以看成一个中转变量

        # 单样本梯度裁剪（每轮固定范数C）
        def microbatch_step(self):  # 一个样本
            total_norm = 0.
           # print("self:", len(self.param_groups))
            # 范数的计算,遍历一个样本梯度中每个元素
            for group in self.param_groups:  # 整个group里面包含params和accum_grads两个模块，params是每层的张量（正常单元和偏执单元分开）
              #  print("microbatch_step-group: ", len(group))
                for param in group['params']:  # param是具体的那个张量（逐层），偏执单元和正常单元的张量分开
                 #   print("microbatch_step-param: ", param.shape)
                    if param.requires_grad:  # 如果可导
                        total_norm += param.grad.data.norm(2).item() ** 2.  #对每层求范数然后把根号开出来，然后对它们求和

            total_norm = total_norm ** .5  #最后求和的数再取平方根，完成了单样本梯度范数的计算
            # print("裁剪toral_norm:",total_norm)
            # print("self.l2_norm_clip:",self.l2_norm_clip)
            clip_coef = min(self.l2_norm_clip / (total_norm+ 1e-6), 1.)  # 范数比较，得到等下要裁剪的部分
            #clip_coef=1.0      #不裁剪
            # 梯度的裁剪
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                 #   print("accum_grad:", accum_grad.shape)
                    if param.requires_grad:
                        # 裁剪是对param裁剪，#add_：参数更新。对accum_grad进行参数更新,将裁剪后的值加到accum_grad中去
                        accum_grad.add_(param.grad.data.mul(clip_coef))  # 单层梯度裁剪，为什么要单层梯度裁剪呢，裁剪范数（范数的计算是所有层的）不变的情况下，其实单层裁剪和全部一起裁剪是一样的，只是单层更具备可调整性

            return total_norm


       #这个是accum_grad清零
        def zero_accum_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()  # 对accum_grad进行梯度清空，因为如果梯度不清零，pytorch中会将上次计算的梯度和本次计算的梯度累加。


        #这里做的是全部样本相加、加噪然后平均
        def step(self, *args, **kwargs):
            for group in self.param_groups:
                # print("group: ",group)
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):  # 整个group里面包含params和accum_grads两个模块 逐层操作
                    # print("param: ",param)
                    if param.requires_grad:

                        # 将accum-grad的值克隆赋值给param.grad
                        param.grad.data = accum_grad.clone()

                        # 对求和的梯度进行加噪。randn_like：返回与输入相同大小的张量，该张量由区间[0,1)上均匀分布的随机数填充。torch.randn_like可以理解为标准正态分布

                        param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))

                        # 再除以batch数平均化
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

            # 调用原函数的梯度下降,假设这个step只做梯度下降，包括学习率的那个梯度更新操作
            super(DPOptimizerClass, self).step(*args, **kwargs)

        # 这里做的是全部样本相加、加噪然后平均
        def step3(self, noise_list, *args, **kwargs):
            i = 0
            for group in self.param_groups:
                # print("group: ",group)
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):  # 整个group里面包含params和accum_grads两个模块
                    i = i + 1
                    # print("param: ",param)
                    if param.requires_grad:
                        # 将accum-grad的值克隆赋值给param.grad
                        param.grad.data = accum_grad.clone()

                        # 对求和的梯度进行加噪。randn_like：返回与输入相同大小的张量，该张量由区间[0,1)上均匀分布的随机数填充。
                        # 我们可以只求一层的，这里其实就是一层，但是这样循环每层都循环进去了。也就是一个a有
                        a = self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data)
                        param.grad.data.add_(a)
                        noise_list.append(a)

                        # 再除以batch数平均化
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)
            # 调用原函数的梯度下降,假设这个step只做梯度下降，包括学习率的那个梯度更新操作
            super(DPOptimizerClass, self).step(*args, **kwargs)

        # 这个step2只做梯度的加噪然后返回加噪后的梯度，不做梯度下降，把这个函数放在每个样本的循环里面，这样就会每次样本对应一个accum_grad
        def step2(self, *args, **kwargs):
            for group in self.param_groups:
                # print("group: ",group)
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):  # 整个group里面包含params和accum_grads两个模块 逐层操作
                    # print("param: ",param)
                    if param.requires_grad:
                        # 将accum-grad的值克隆赋值给param.grad
                        param.grad.data = accum_grad.clone()

                        # 对求和的梯度进行加噪。randn_like：返回与输入相同大小的张量，该张量由区间[0,1)上均匀分布的随机数填充。torch.randn_like可以理解为标准正态分布
                        param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))

                        # 不用平均，只有一个样本
                       # param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

        #下面的函数针对自适应范数的
        #应用Abail的建议，取本轮batch所有梯度的范数的中位数作为C,这里执行范数计算但不裁剪
        def compute_L2Norm(self):  # 一个样本
            total_norm = 0.
            Ada_norm=0.
            perlayer_norm=0.
            C=0.       #定义裁剪范数

            # print("self:", len(self.param_groups))
            # 范数的计算,遍历一个样本梯度中每个元素
            for group in self.param_groups:  # 整个group里面包含params和accum_grads两个模块，params是每层的张量（正常单元和偏执单元分开）
                #print("microbatch_step-group: ", len(group))             len(group)是所有层的层数，包括偏置层
                for param in group['params']:  # param是具体的那个张量（逐层），偏执单元和正常单元的张量分开
                    #   print("microbatch_step-param: ", param.shape)
                    if param.requires_grad:  # 如果可导
                        total_norm += param.grad.data.norm(2).item() ** 2.  #对每层求范数然后把根号开出来，然后对它们求和
            total_norm = total_norm ** .5  # 最后求和的数再取平方根，完成了单样本梯度范数的计算
            # print("计算toral_norm:",total_norm)
            return total_norm

        #计算单样本每层梯度的范数形成一个数组
        def compute_L2Norm_perlayer(self):  # 一个样本
            perlayer_norm=np.zeros(8,dtype=float,order='C')  #8是网络模型层数
            i=0

            # print("self:", len(self.param_groups))
            # 范数的计算,遍历一个样本梯度中每个元素
            for group in self.param_groups:  # 整个group里面包含params和accum_grads两个模块，params是每层的张量（正常单元和偏执单元分开），这里循环的是每层
                #print("microbatch_step-group: ", len(group))             len(group)是所有层的层数，包括偏置层
                for param in group['params']:  # param是具体的那个张量（逐层），偏执单元和正常单元的张量分开,这里循环的是每个神经元
                    #   print("microbatch_step-param: ", param.shape)
                    if param.requires_grad:  # 如果可导
                        perlayer_norm[i]=param.grad.data.norm(2).item() ** 2. #对每层求范数然后把根号开出来，然后对它们求和。得到每层的范数，还没有根号的
                i+=1
            perlayer_norm=perlayer_norm** .5   #得到该样本每层的范数的数组
            return perlayer_norm

        #单个样本逐层不同范数裁剪,完成裁剪的同时返回这轮的梯度的范数
        def microbatch_step_perlayer(self):
            total_norm = 0.
            perlayer_norm=np.zeros(8,dtype=float,order='C')
            i=0
            j=0
            clip_coef_perlayer=np.zeros(8,dtype=float,order='C')

            # print("self:", len(self.param_groups))
            # 范数的计算,遍历一个样本梯度中每个元素
            for group in self.param_groups:  # 整个group里面包含params和accum_grads两个模块，params是每层的张量（正常单元和偏执单元分开），这里循环的是每层
                #print("microbatch_step-group: ", len(group))             len(group)是所有层的层数，包括偏置层
                for param in group['params']:  # param是具体的那个张量（逐层），偏执单元和正常单元的张量分开,这里循环的是每个神经元
                    #   print("microbatch_step-param: ", param.shape)
                    if param.requires_grad:  # 如果可导
                        perlayer_norm[i]=param.grad.data.norm(2).item() ** 2.  #得到每层的梯度平方和
                    i+=1

            perlayer_norm=perlayer_norm** .5 #得到该样本每层的范数
            for i in range(0,8):
                clip_coef_perlayer[i] = min(self.l2_norm_clip[i] / (perlayer_norm[i] + 1e-6), 1.)  # 范数比较，得到等下每层要裁剪的部分，这边每层裁剪的范数已经在上一轮保存在l2_norm_clip中

            # 逐层的梯度的裁剪
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']): #逐层
                    #   print("accum_grad:", accum_grad.shape)
                    if param.requires_grad:  #逐个变量
                        # 裁剪是对param裁剪，#add_：参数更新。对accum_grad进行参数更新,将裁剪后的值加到accum_grad中去，那这个accum_grad是什么
                        accum_grad.add_(param.grad.data.mul(
                            clip_coef_perlayer[j]))  # 单层梯度裁剪
                    j=j+1


            return perlayer_norm

        # 这里进行逐层加噪
        def step_perlayeraddnoise(self, *args, **kwargs): #这边用的是上一轮的每层的梯度的范数的平均值进行每层的范数进行加噪，裁剪范数存储在
            i=0
            for group in self.param_groups:
                # print("group: ",group)
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):  # 整个group里面包含params和accum_grads两个模块
                    # print("param: ",param)
                    if param.requires_grad:
                        # 将accum-grad的值克隆赋值给param.grad
                        param.grad.data = accum_grad.clone()  #accum_grad是model中内在的一个变量，每层中的每个梯度都已经累加好了

                        # 对求和的梯度进行加噪。randn_like：返回与输入相同大小的张量，该张量由区间[0,1)上均匀分布的随机数填充。

                        #每层因为裁剪的范数不同，所有加噪时候用的范数也应该不同，sigma要乘以根号的层数，这样才能保持同样的隐私效果,
                        param.grad.data.add_(self.l2_norm_clip[i] *np.sqrt(len(group))* self.noise_multiplier * torch.randn_like(param.grad.data)) #每层分别加噪

                        # 再除以batch数平均化
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)
                    i=i+1
            # 调用原函数的梯度下降,假设这个step只做梯度下降，包括学习率的那个梯度更新操作
            super(DPOptimizerClass, self).step(*args, **kwargs)


    return DPOptimizerClass

#括号后面的是从Pytorch optimizer中调用的梯度下降函数，然后make_optimizer_class是自己对后面调用的原函数进行封装，如上
DPAdam = make_optimizer_class(Adam)
DPAdagrad = make_optimizer_class(Adagrad)
DPSGD = make_optimizer_class(SGD)
DPRMSprop = make_optimizer_class(RMSprop)


