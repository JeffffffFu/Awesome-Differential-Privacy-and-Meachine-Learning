

<!-- [![Stars](https://img.shields.io/github/stars/youngfish42/Awesome-Federated-Learning-on-Graph-and-Tabular-Data.svg?color=orange)](https://github.com/youngfish42/Awesome-Federated-Learning-on-Graph-and-Tabular-Data/stargazers)  [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re) [![License](https://img.shields.io/github/license/youngfish42/Awesome-Federated-Learning-on-Graph-and-Tabular-Data.svg?color=green)](https://github.com/youngfish42/image-registration-resources/blob/master/LICENSE) ![](https://img.shields.io/github/last-commit/youngfish42/Awesome-Federated-Learning-on-Graph-and-Tabular-Data) -->

Person Website（个人主页）: https://Jefffffffu.github.io/ 

bilibili（论文视频分享）:https://space.bilibili.com/80356866/video

关于DP-FL更完整更系统的总结，请查阅：https://arxiv.org/abs/2405.08299

推荐一个DP社区：https://differentialprivacy.org/

# Citations
以上的分类总结和代码来自于以下论文，如果你从以上总结得到了帮助或者你有使用相关代码，请您在文章中引用以下论文。这对我非常重要，谢谢。
```bash
@inproceedings{fu2022adap,
  title={Adap dp-fl: Differentially private federated learning with adaptive noise},
  author={Fu, Jie and Chen, Zhili and Han, Xiao},
  booktitle={2022 IEEE International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom)},
  pages={656--663},
  year={2022},
  organization={IEEE}
}
```
```bash
@article{fu2024differentially,
  title={Differentially private federated learning: A systematic review},
  author={Fu, Jie and Hong, Yuan and Ling, Xinpeng and Wang, Leixia and Ran, Xun and Sun, Zhiyu and Wang, Wendy Hui and Chen, Zhili and Cao, Yang},
  journal={arXiv preprint arXiv:2405.08299},
  year={2024}
}
```

# Papers Organization
**Table of Contents**

- [Papers](#Papers)
  - [DP Theory](#dp-theory)  
    - [Differential Adversary Definition](#differential-adversary-definition)
      - [CDP](#cdp)  
      - [LDP](#ldp) 
    - [Privacy Measurement Method](#privacy-measurement-method)
      - [DP](#dp)
      - [RDP(MA)](#rdpma)
      - [ZCDP](#zcdp)
      - [GDP](#gdp)
      - [Bayesian DP](#Bayesian-DP)
    - [Privacy Amplification Technology](#privacy-amplification-technology)
      - [Sampling](#samlping)
      - [Shuffle](#shuffle)
  - [DP and Meachine Learning](#dp-and-meachine-learning)  
    - [Meachine Leaning](#meachine-learning)
    - [Meachine Leaning with DP](#meachine-learning-with-dp)
    - [GNN](#gnn)
    - [GNN with DP](#gnn-with-dp)
    - [Privacy of GNN](#privacy-of-gnn)
  - [Federated Learning](#federated-leaning)
  - [Differentlly Private FL](#differentilly-private-fl)
    - [HFL](#hfl)
      - [CDP-HFL](#cdp-hfl)
        - [SL-DP](#sl-dp)
        - [CL-DP](#cl-dp)
      - [LDP-HFL](#ldp-hfl)
      - [shuffle model-HFL](#shuffle-model-hfl)
        - [Shuffle model with SL-DP](#shuffle-model-with-sl-dp)
        - [Shuffle model with CL-DP](#shuffle-model-with-cl-dp)
    - [VFL](#vfl)
    - [TFL](#tfl)
    - [incentive](#incentive)
  - [Attack](#attack)
    - [MIA](#membership-inference-attack)
  - [Application Scenarios](#application-scenarios)
    - [Text Protection](#text-protection)
    - [Recommended System](#recommended-system)
    - [DP and image](#dp-and-image)
    - [DP and cypto](#dp-and-cytro)
  - [Meachine Unlearning](#meachine-unlearning)
    - [Unlearning in Centralized machine learning](#unlearning-in-centralized-machine-learning)
    - [Unlearning in FL](#unlearning-in-fl)

## DP Theory

### Differential Adversary Definition
CDP（central DP）有一个完全可信的中心方，敌手是外界。而LDP（local DP）认为中心方是诚实但好奇的。
#### CDP
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| Differential privacy | Cynthia Dwork | ICALP/2006 | 首次提出差分隐私的定义 | 
| Programming Differential Privacy （Book）| Joseph P. Near and Chiké Abuah | 2021 | 讲诉了DP的概念定理和机制等，并附有简单代码呈现（简单入门推荐）[【Link】](https://programming-dp.com/) | 
| The Algorithmic Foundations of Differential Privacy（Book） | Cynthia Dwork | 2014 | DP的定义理论，高级组合和相关机制等的完整证明推导（更加理论）[【拉普拉斯、严格差分、高斯机制、松弛差分】](https://www.bilibili.com/video/BV18r4y1j7Bs?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3) |
 | Differential Privacy From Theory to Practice （Book）| Ninghui Li | 2014 | 除了一些基本定理和机制，用了具体的实际例子讲诉了DP的用法及DP的伦理探讨（更加实用化）[【Chapter1、Chapter2】](https://www.bilibili.com/video/BV1br4y1J7Qn?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3),[原作者讲解]()|

#### LDP
| Title                                                                                        | Team/Main Author       | Venue and Year                                           | Key Description                                                                                                                                                                                                               
|:---------------------------------------------------------------------------------------------|:-----------------------|:---------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| 本地化差分隐私研究综述                                                                                  | 中国人民大学                 | Journal of Software/2018                                 | 介绍了本地化差分隐私的原理与特性,总结和归纳了LDP当前研究工作,重点阐述了频数统计、均值统计的LDP机制设计[【vedio】](https://www.bilibili.com/video/BV18B4y1a75b?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)                                              | 
| RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response                          | Google                 | ACM SIGSAC /2014                                         | 1.RAPPOR的分类和对应扰动方法。2.Differential Privacy loss of RAPPOR（主要是多维）                                                                                                                                                               | 
| Locally Differentially Private Protocols for Frequency Estimation                            | Purdue University      | USENIX/2017                                              | 1.提出了一个Pure LDP Protocols，并基于Pure LDP Protocols给出了其方差和频数估计的公式。2.提出OUE（UE=basic RAPPOR），给出了q的最佳选择和方差。[【vedio】](https://www.bilibili.com/video/BV1HW4y127mp?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3) | 
| Collecting High-Dimensional and Correlation-Constrained Data with Local Differential Privacy | Rong Du                | sensor, mesh and ad hoc communications and networks/2021 | 该文章重要的是针对LDP的高维度情况，本篇重点在于推断多维RR的概率关系： 1、进行属性之间相关性的度量（首先会属性会分为不同的簇，簇之间独立不相干，簇之内的属性相关），定义了p这个变量定义来定理相关性（基于协方差） 2、提出UDLDP，基于以上在单个属性上定义LDP，并提出CBP                                                                                 |                                                       |                                                                                                                                                                                                                              |
| Collecting and Analyzing Data from Smart Device Users with Local Differential Privacy        | Thông T. Nguyên et al. | arxiv/2016                                               | 提出了Harmony，用于包含数值和类别属性的多维数据的LDP下的均值和频数统计。主要是连续型数据直接随机的对称扰动成两个相反数，然后保证均值无偏，误差边界比用Lap小。                                                                                                                                         |                                                                                                                                                                                                                              |
| Collecting and Analyzing Multidimensional Data with Local Differential Privacy               | Ning Wang et al.       | ICDE/2019                                                | 核心是单维数据LDP的收集，多维是一个简单扩展。单维下文章发现了拉普拉斯加噪和DM（Duchi）的优缺点，两个方法随着eps的增大有一个交点。文章结合两个方法的优点提出PM方法，随后引入alpha参数升级为HM得到更小的误差边界 [【vedio】](https://www.bilibili.com/video/BV1u8411J7he/?spm_id_from=333.999.list.card_archive.click&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)                                                                                                            |                                                                                                                                                                                                                              |
| A Comprehensive Survey on Local Differential Privacy               | Wuhan University      | Security and Communication Networks/2020                 | 目前看起来较为新较为全面的LDP综述，对于LDP的各种机制和应用都有概括。可以反复进行参考                                                                                                                                                                                 |                                                                                                                                                                                                                              |
| Utility-Optimized Local Differential Privacy Mechanisms for Distribution Estimation               | Takao Murakami      | USENIX Security/2019                                      | 针对LDP下的不同回答（属性），文章认为有些是敏感的需要保护，有些是不敏感的不需要保护，由此提出一种新的LDP叫ULDP，ULDP使得在同样的预算下做更少的扰动，使得方差误差更小。[【vedio】](https://www.bilibili.com/video/BV17D4y1K7mS/?spm_id_from=333.999.list.card_archive.click&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)                                                                                                                                    |                                                                                                                                                                                                                              |

### Privacy Measurement Method

#### DP
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| Differential privacy | Cynthia Dwork | ICALP/2006 | 首次提出差分隐私的定义 | 
| Programming Differential Privacy （Book）| Joseph P. Near and Chiké Abuah | 2021 | 讲诉了DP的概念定理和机制等，并附有简单代码呈现（简单入门推荐） | 
| The Algorithmic Foundations of Differential Privacy（Book） | Cynthia Dwork | 2014 | DP的定义理论，高级组合和相关机制等的完整证明推导（更加理论）[【拉普拉斯、严格差分、高斯机制、松弛差分】](https://www.bilibili.com/video/BV18r4y1j7Bs?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3) |
| Differential Privacy From Theory to Practice （Book）| Ninghui Li | 2014 | 除了一些基本定理和机制，用了具体的实际例子讲诉了DP的用法及DP的伦理探讨（更加实用化）[【Chapter1、Chapter2】](https://www.bilibili.com/video/BV1br4y1J7Qn?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3),[原作者讲解]()|
| The Bounded Laplace Mechanism in Differential Privacy| IBM | 2018 | 提出了阈值限定的重加噪拉普拉斯机制无法满足纯DP，但是可以满足松弛DP，文章给出了噪声系数需要满足一个式子(该式子和区间，eps，delta相关)即可满足(eps,delta)-DP。这边的松弛DP不是用的RDP，而是直接在后面加一个delta来放缩sigma。|

#### RDP(MA)
此前的MA（monments accountant）目前来看就是RDP，区别在于RDP从一开始的散度就用了高阶矩进行度量，而MA是在分布上进行高阶矩度量，不过最后的表达形式几乎一致，两者思想也一致。

| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| Deep Learning with Differential Privacy | Martín Abadi |  CCS/2016 | 首次提出高斯下采样机制并利用Moments accountant技术进行隐私损失度量 [【vedio】](https://www.bilibili.com/video/BV1r44y1u7iP?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)| 
| Rényi Differential Privacy | Ilya Mironov |  Security Foundations Symposium/2017 | Rényi 差分隐私的定义、性质和结论。主要关注里面的高斯机制（无下采样），组合性质及转为DP的公式。[【vedio】](https://www.bilibili.com/video/BV1YF411L7xV?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3) | 
| Subsampled Rényi Differential Privacy and Analytical Moments Accountant | Yu-Xiang Wang | AISTATS /2018 | RDP和MA的直观对比，主要给出了一个RDP高斯下采样的一个隐私损失上界，有很多论文引用这个上界的结论。 | 
| Rényi Differential Privacy of the Sampled Gaussian Mechanism | Ilya Mironov | 2019 | 给出了Rényi Differential Privacy of the Sampled Gaussian Mechanism 各种情况的隐私损失公式，里面的3.3的公式目前应用于各个开源库的隐私计算中（更为简洁） |
| Hypothesis Testing Interpretations and Rényi Differential Privacy | Balle B | AISTATS/2020 | 定理21给出了更为紧凑的RDP转DP的公式（目前开源库opacus应用的是这个转换）  | 

#### ZCDP
| Title | Team/Main Author | Venue and Year | Key Description                                                                                                                              
| :------------| :------ | :---------- |:---------------------------------------------------------------------------------------------------------------------------------------------
| Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds | UMark Bun | 2016 | 提出ZCDP的定义及ZCDP和DP的关系（ZCDP也是从瑞丽散度来定义的）。但是zcdp和RDP具体定义上还有些不同。无论alpha取什么值，zcdp都是小于等于RDP的。[RDP与ZCDP对比参考](https://zhuanlan.zhihu.com/p/457972991) | 
| Differentially Private Model Publishing for Deep Learning | Gatech | SP/2019| 提出了ZCDP的采样和随机排列机制                                                                                                                            | 

#### GDP
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| Gaussian Differential Privacy | University of Pennsylvania |  2019 | 首次提出一种基于假设建议的新的新的差分隐私-GDP，文章包括GDP的定义，性质和机制用法等 [【vedio】](https://www.bilibili.com/video/BV1xd4y1D7UV?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)[原作者讲解](https://www.bilibili.com/video/BV1SY4y1b73T?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)| 
| Deep Leaning with Gaussian Differential Privacy| University of Pennsylvania | 2020 | 提炼了上面的Gaussian Differetial Privacy的结论公式，区别在于给出了一个基于泊松采样的GDP | 
| Federated f-Differential Privacy | University of Pennsylvania | AISTATS/2021 | 样本级的差分隐私保护，提出了强联邦和弱联邦（个人认为弱联邦没什么意义），隐私度量采用GDP。 | 
| DISTRIBUTED GAUSSIAN DIFFERENTIAL PRIVACY VIA SHUFFLING | University of Pennsylvania | ICLR(workshop)/2021 | 采用f-dp去证明shuffle模型，在f-dp处主要求第一类错误率和第二类错误率，然后得到一个最大的u。在shuffle中就去找这两类错误率，文中的shuffle证明思路采用Cheu等人的shuffle证明，从布尔求和及实数求和两个场景去证明shuffle满足的差分隐私。[【vedio】](https://www.bilibili.com/video/BV1et4y1g7Dv?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3) | 

#### Bayesian DP
TO DO

《Data-Aware Privacy-Preserving Machine Learning》
### Privacy Amplification Technology

#### samlping
目前的采样还是从CDP的角度出发衡量的

| Title                                                                                                | Team/Main Author | Venue and Year | Key Description                                                                                             
|:-----------------------------------------------------------------------------------------------------|:-----------------|:---------------|:------------------------------------------------------------------------------------------------------------
| On Sampling, Anonymization, and Differential Privacy: Or, k-Anonymization Meets Differential Privacy | Purdue University       | 2011           | 给出了最原始的采样隐私预算公式和证明                                                                                          |
| Rényi Differential Privacy of the Sampled Gaussian Mechanism                                         | Ilya Mironov     | 2019           | 给出了Rényi Differential Privacy of the Sampled Gaussian Mechanism 各种情况的隐私损失公式，里面的3.3的公式目前应用于各个开源库的隐私计算中（更为简洁） |
| Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences | Borja Balle     | 2019           | 给出了均匀采样无放回，均匀采样有放回和泊松采样的三种采样方式的隐私放大公式 |

#### shuffle
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ |:---------------| :----------------------- 
| The Privacy Blanket of the Shuffle Model | Borja Balle  | 2019           | 最为经典的shuffle证明。首先指出，对于shuffle模型，整个shuffle中的数据集应当看成整体，从而才有相邻数据集，针对这个数据集整体满足差分隐私。其次，采用数据相关和数据无关将整体数据分割成两部分，最大化敌手，使得最后只需要分析数据无关的部分，即隐私毯子，随机性在隐私毯子中。[【vedio】](https://www.bilibili.com/video/BV14W4y1b7VK?spm_id_from=333.999.list.card_archive.click&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)
| Hiding Among the Clones: A Simple and Nearly Optimal Analysis of Privacy Amplification by Shuffling | Vitaly Feldman  | arxiv/2020     | 和隐私毯子类似的证明思想，不过假设的敌手知道的背景知识没有隐私毯子那么全面，每一个扰动的输出都会影响隐私性。在数据相关部分，进一步分成p/2和p/2进行分析。得到的隐私界比隐私毯子更紧凑。
| ESA:A Nove lPrivacy Preserving Framework | Renmin University of China  | Journal of Computer Research and Development /2021 | 系统的介绍ESA框架（基于shuffle的LDP）及应用                                                                                                                                                                                                                                                                  | 
| Shuffle Gaussian Mechanism for Differential Privacy | Seng Pei Liew               | 2022                                               | 本地客户端进行高斯加噪后得到一个梯度，然后对不同客户端的梯度进行shuffle进行隐私放大。并用不放回随机采样和泊松采样的高斯机制结合shuffle进行理论证明（基于RDP）。                                                                                                                                                                                                      
| The Privacy Blanket of the Shuffle Model | Borja Balle                 | 2019                                               | 最为经典的shuffle证明。首先指出，对于shuffle模型，整个shuffle中的数据集应当看成整体，从而才有相邻数据集，针对这个数据集整体满足差分隐私。其次，采用数据相关和数据无关将整体数据分割成两部分，最大化敌手，使得最后只需要分析数据无关的部分，即隐私毯子，随机性在隐私毯子中。[【vedio】](https://www.bilibili.com/video/BV14W4y1b7VK?spm_id_from=333.999.list.card_archive.click&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3) 
 
## DP and Meachine Learning
本章集中在如何进行更高效机器学习集合差分隐私训练
### Meachine Learning
| Title | Team/Main Author | Venue and Year | Key Description                                                                                                                                                                                                 
| :------------| :------ |:---------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation | Sebastian Bach | PLOS ONE/2015  | 利用LRP（分层相关传播）算法，反向求出每个神经网络中每个元素的相关性贡献和输入像素对预测函数f的贡献                                                                                                                                                             | 
| The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks | Jonathan Frankle | ICLR/2019      | 文章认为每一个大网络中藏着一个小的子网络可以表达模型，称为中奖网络。先初始化模型然后训练到一定精度，按照网络中每个神经元权值大小排序前百分之X得到一个mask，即子网络，然后从一开始的初始化模型上重新用这个子网络训练。                                                                                                   | 
| Prospect Pruning: Finding Trainable Weights at Initialization using Meta-Gradients | Oxford | ICLR/2022      | 文章提出前景剪枝，一开始初始化一个全为1的mask，带着mask进行几轮训练，然后反向得到梯度，再用梯度对mask求导，得到每个mask对应的梯度，文章称为元梯度。然后对这些mask上的元梯度按照值的大小进行排序，进而确定最后的mask。该方法预训练只需要几步即可确定，而且是oneshot的剪枝。|


### Meachine Learning with DP
| Title                                                                                           | Team/Main Author                | Venue and Year                                      | Key Description                                                                                                                                                                                                                                       
|:------------------------------------------------------------------------------------------------|:--------------------------------|:----------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| Deep Learning with Differential Privacy                                                         | Martín Abadi                    | CCS/2016                                            | 首次将差分隐私和深度学习梯度下降结合提出DPSGD算法 [【vedio】](https://www.bilibili.com/video/BV1r44y1u7iP?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)                                                                                                 | 
| Differentially Private Model Publishing for Deep Learning                                       | Gatech                          | SP/2019                                             | 1.提出了ZCDP的下采样机制。2.提出了自适应的噪声系数下降                                                                                                                                                                                                                       | 
| Concentrated Differentially Private Gradient Descent with Adaptive per-Iteration Privacy Budget | University of Georgia           | ACM SIGKDD/2018                                     | 基于梯度的变化来自适应进行隐私预算分配，其实就是根据梯度的变化来进行噪声系数的自适应变化，思想是梯度大的时候多加噪，梯度小的时候少加噪                                                                                                                                                                                   | 
| An Adaptive and Fast Convergent Approach to Differentially Private Deep Learning                | University of Nanjing           | INFOCOM/2020                                        | 1.在原有的梯度下降法RMSPROP上加上DP。2.提出了梯度中逐元素的自适应裁剪（裁剪阈值的自适应，每个梯度元素对应一个裁剪阈值），这个裁剪值根据RMSPROP方法中每轮每个元素积累的值来衡量 [【vedio】](https://www.bilibili.com/video/BV1qr4y1M7sq?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)                           | 
| Tempered Sigmoid Activations for Deep Learning with Differential Privacy                        | Nicolas Papernot                | AAAI/2021                                           | 在原有的DPSGD上，对于神经网络结构中的激活函数用缓和的sigmod函数-tanh（有界函数）来替代Relu（无界函数），有界函数可以使得梯度控制在某个阈值中，从而让裁剪范数C不会大于梯度范数值。                                                                                                                                                   | 
| Semi-Supervied Knowledge Transfer For Deep Learning From Private Training Data                  | Nicolas Papernot                | ICLR/2017                                           | 提出了差分隐私结合机器学习的新范式，构建teacher model（私有数据集）的标签投票作为输出的DP query，利用标签也就不可避免得引入了knowledge transfer的方式来将student model（公共数据集）作为原模型的影子[【vedio】](https://www.bilibili.com/video/BV1W34y1775e?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3) | 
| Three Tools for Practical Differential Privacy                                                  | Koen Lennart van der Veen       | Machine Learning/2018                               | 提出三种提高DPSGD精度的方法：1、训练前进行完整性检查。2、进行分层裁剪（没给隐私度量方式）3、大批量训练（每次训练采样率加大）                                                                                                                                                                                    | 
| Differentially Private Learning Needs Better Features (or Much More Data)                       | Stanford                        | ICLR/2021                                           | 该论文以Nicolas的tanh优化为baseline进行比对，提出了人工的特征提取（handcrafted features）的方式，用了ScatterNet先进行特征提取，分别用ScatterNet+Linear model和ScatterNet+CNN都有很好的效果，目前看起来是最高的准确率（但是添加了预处理的步骤），该论文有代码，提供了相关超参数                                                                    | 
| Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger               | AWS AI                          | ICLR/2022                                           | 该论文以Nicolas的tanh优化为baseline进行比对，提出了最小化裁剪范数，然后整个裁剪加噪后的式子可以简化成裁剪范数和学习率的线性关系。作者提出只要取一个很小的裁剪范数，然后调整学习率就可以获得高的准确率，即固定一个很小的裁剪范数即可（范数裁剪不会影响梯度方向，因为是整体线性裁剪）。后面作者加了一个gamma在裁剪的时候使得收敛性得以证明，也比之前的baseline效果好一点点。这篇文章的结合DP的收敛性分析值得借鉴。                             | 
| DPNAS: Neural Architecture Search for Deep Learning with Differential Privacy                   | Chinese Academy of Sciences     | national conference on artificial intelligence/2021 | 这篇论文是寻找最适合DP深度学习的网络结构，文章会在每层都提供一些网络设计，然后遍历的去尝试，哪种组合会更好，这篇文章实现的准确率比上一篇还高。他们发现了SELU激活函数表现比tanh好。                                                                                                                                                        | 
| Label differential privacy via clustering                                                       | Google                          | arxiv/2021                                          | 提出了标签差分隐私，即对标签进行KRR扰动满足差分，但是在对标签扰动之前先进行聚类，使得相同分布的样本标签在同一簇中，聚类的过程有进行差分隐私加噪在标签上。然后簇中进行样本标签的扰动。最后在loss函数中行乘以对应扰动机制的逆进行矫正。以此完成图像分类任务。                                                                                                                     | 
| Local Differential Privacy for Deep Learning                                                    | RMIT University | IEEE Internet of Things/2020                        | 本篇文章给出了在下采样和池化层之后，对数据特征进行扰动，在进行全连接层之前进行扰动。比较核心在于对整个梯度进行编码，利用十进制转二进制，在进行比特的扰动。后文引入的优化方法的参数a存在偷换概念的问题，相当于引入了一个新的隐私预算，但是缺忽视掉了。                                                                                                                           | 
| Differential Privacy Meets Neural Network Pruning                                                    | Kamil Adamczewski                           | arxiv/2023                                          | 文章将传统按权值剪枝的技术和DPSGD进行结合，确定剪枝的mask是通过公共数据集得到的。进行DPSGD结合有两点好处，一是在进行裁剪的时候，只对参与训练的神经元进行裁剪，也就是减少了裁剪幅度，也相当于double clip。其二加噪声时，只对参与训练的神经元加噪，这样减少了噪声对模型的影响，尤其在大模型下。                                                                                           | 

### GNN 
| Title                                                                                                                                  | Team/Main Author                      | Venue and Year      | Key Description                                                           
|:---------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------|:--------------------|:--------------------------------------------------------------------------
| Federated Graph Machine Learning: A Survey of Concepts, Techniques, and Applications                                                        | University of Virginia                      | KDD/2022            | 图联邦学习综述，对于FL结合graph三种异质情况的分类：1、客户端之间点重合；2、客户端之间边缺失；3、客户端之间数据异质（这个和传统联邦一致） | 
| Graph Sparsification via Meta-Learning                                                                                                 | Graph Sparsification via Meta-Learning| ICDE(workshop)/2021 | 在节点分类任务中，通过对邻接矩阵求导来进行邻接矩阵的稀疏化，对应边为1的地方，求导梯度值越大，越应该置为0                     | 

### GNN with DP
| Title                                                                                        | Team/Main Author  | Venue and Year      | Key Description                                                                                                      
|:---------------------------------------------------------------------------------------------|:------------------|:--------------------|:---------------------------------------------------------------------------------------------------------------------
| GAP: Differentially Private Graph Neural Networks with Aggregation Perturbation              | Sina Sajadmanesh  | USENIX/2023         | 在非端到端的图机器学习中，提出边级别和节点级别的图差分隐私。其主要想法是在消息传递，即特征聚合的时候利用CDP的概念进行加噪，在聚合前进行特征归一化从而获得每个节点特征的l2范数的bound，以获得敏感度。              | 
| Node-Level Differentially Private Graph Neural Networks                                      | Google            | ICLR(workshop)/2022 | 将DPSGD应用到图的节点分类任务，主要是限制图的最大度，从而利用N叉树的总节点个数去算出每个节点在K跳的情况下影响到多少个节点的gradient，从而确定敏感度。并有一个针对这个场景的新的采样放大的隐私预算推导           | 
| SoK: Differential Privacy on Graph-Structured Data                                           | T. Mueller et al. | Arxiv/2022          | 图结合DP的综述文章                                                                                                           |
| PrivGraph: Differentially Private Graph Data publication by Exploiting Community Information | ZJU               | USENIX/2023         | 是一个边DP的图生成工作，其利用了LM先进行社区划分（社区划分的时候对初的社区信息添加LAP），后面根据社区间的信息和社区内的信息进行图生成，借助了CL model进行重生成。重生成之前对社区间的信息和社区内的信息添加LAP进行扰动 | 

### Privacy of GNN
| Title                                                                                  | Team/Main Author | Venue and Year | Key Description                                                                                                                                               
|:---------------------------------------------------------------------------------------|:-----------------|:---------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------
| Disparate Vulnerability in Link Inference Attacks against Graph Neural Networks        | Stevens          | PETS/2023      | 提出了图中组（针对标签）密度不均匀会加大敏感边的泄露的问题。假设有A，B两种节点标签，这里就存在三组边的联系，A和A，A和B，B和B。A-A组的密度定义为实际这个图中A-A连接的边数除以全部A-A有可能的边数。文章的idea是在训练GNN的过程中，通过扰动（加边或删边）的方式让各组的密度差异不大，进而保护边隐私。 | 
| linkteller: recovering private edges from graph neural networks via influence analysis | UIUC             | S&P/2022       | 构建了一个纵向图联邦的场景，一方持有特征和标签（public），一方持有边(private)。提出了一种攻击方式，每次改变一个节点的特征观察GNN的输出，去判断节点之间连接关系，本质是一种差分攻击。文章后面提出了用DP进行保护，分别是lap和RR。                                  | 
| Group property inference attacks against graph neural networks                         | Stevens          | CCS/2022       | 提出了节点的组标签攻击。比如一个图中有两种标签：男和女。通过攻击揭露这两种标签在图中的占比。                                                                                                                | 
| GraphGAN: Graph Representation Learning with Generative Adversarial Nets               | SJTU             | AAAI/2017      | 基于生成对抗网络进行图的边的重生成，其目标是生成的边和原来的边尽量的相同。生成器会随机选择某几个节点作为该节点的邻居，而判别器给这些邻居节点打分，基于该节点附近选择的真实邻居作为评判标准来鉴别。生成器和判别器同时优化一个损失函数                                            | 



## Federated Leaning
想看更多的联邦学习文献推荐可以转到 :

[Awesome-Federated-Learning-on-Graph-and-GNN-papers](https://github.com/huweibo/Awesome-Federated-Learning-on-Graph-and-GNN-papers) 

 [Awesome-Federated-Machine-Learning](https://github.com/innovation-cat/Awesome-Federated-Machine-Learning#16-graph-neural-networks)

 [个性化联邦学习代码仓](https://github.com/microsoft/PersonalizedFL)

| Title                                                                                            | Team/Main Author                 | Venue and Year                                           | Key Description                                                                                                                                                                                                                                                            
|:-------------------------------------------------------------------------------------------------|:---------------------------------|:---------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| A Survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection | NUS                              | IEEE Transactions on Knowledge and Data Engineering/2021 | 从各个维度对联邦学习进行总结分类，还有联邦学习主流开源库的对比，是比较完善全面的联邦学习综述                                                                                                                                                                                                                             | 
| Communication-efficient learning of deep networks from decentralized data                        | H Brendan McMahan                | AISTATS/2017                                             | 首次提出了FedAvg算法，即本地迭代多轮再参与联邦平均，可以大大的节省通讯成本                                                                                                                                                                                                                                   | 
| Federated Optimization in Heterogeneous Networks                                                 | University of Pennsylvania       | Proceedings of Machine Learning and Systems/2020         | 针对Non-iid场景提出FedProx算法，与FedAvg的区别就是在客户端本地的损失函数上加了正则项（同时考虑中心方模型）防止本地迭代过拟合                                                                                                                                                                                                   | 
| Adaptive Federated Learning in Resource Constrained Edge Computing Systems                       | Shiqiang Wang                    | JSAC/2019                                                | 在本地迭代资源与中心方聚合资源受限的情况下，最优化本地迭代次数，即最优化客户端参与中心方聚合的频率（文章的引言讲到如果不考虑资源受限，本地迭代一次就参与联邦是最好的，这里的迭代一次应该是一次epoch，需要训练完所有样本，不训练完本地所有样本可能这个结论不成立）                                                                                                                                        | 
| Layer-wised Model Aggregation for Personalized Federated Learning                                | Hong Kong Polytechnic University | CVPR/2022                                                | 中心方的加权平均上对每个客户端的模型每层的参数对实现了个性化的加权，权重的生成利用超网络，对每个客户端每层参数进行再学习，用每个客户端的模型和上一轮中心方模型的差当作损失值进行最小化。后面进一步为了降低通讯成本，选择本地保留权重大的层不参与联邦。 [【vedio】](https://www.bilibili.com/video/BV1P8411L7EE/?spm_id_from=333.999.list.card_archive.click&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3) |                                                                                    | 
| Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization                               | Carnegie Mellon University       | NIPS/2020                                                | 考虑了不同客户端的算力不同，假设客户端存在异质的算力资源（即不同的本地迭代次数的能力）提出FedNova算法。                                                                                                                                                                                                                    |                                                                                    | 

### PFL
一种个性化联邦学习，学习的目标在于各个客户端学到一个在本地数据集上表现效果好的模型。

| Title                                                                                            | Team/Main Author                 | Venue and Year                                           | Key Description                                                                                                                                                                                                                                                       
|:-------------------------------------------------------------------------------------------------|:---------------------------------|:---------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| Ditto: Fair and robust federated learning through personalization                               | T Li                             | ICML/2021                                                | 个性化的联邦学习（以本地客户端数据作为性能测试标准），其中引入公平性和鲁棒性的指标。算法在传统的FedAvg的基础上每个客户端多了本地模型的训练，本地模型的训练在损失函数上加了本地模型和全局模型的差作为正则化项。主要训练出每个客户端的本地模型。                                                                                                                                                 |                                                                                    | 

                                                                                                                                        


## Differentilly Private FL


### HFL

#### CDP-HFL
DP(CDP)的保护理念在于少一个或多一个数据（这里的数据可以包括：样本，用户，客户端）。DP必须存在相邻数据集的概念
##### SL-DP
保护客户端下样本的参与信息，每个客户端下的样本看成一个数据，将联邦中心方看成敌手。基于少一个或多一个样本下的DP保护机制。本质就是本地客户端进行DPSGD算法。特点在于本地客户端每轮迭代进行逐个样本裁剪获得敏感度，然后对梯度之和进行加噪。

| Title                                                                                                         | Team/Main Author | Venue and Year | Key Description                                                                                                                                                                                                                                        
|:--------------------------------------------------------------------------------------------------------------| :------ | :---------- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| DP-FL: a novel differentially private federated learning framework for the unbalanced data                    | Huang et al.| WWW/2020 | 将DPAGD-CNN算法集合联邦机器学习，DPAGD-CNN是集中式机器学习下差分隐私预算的自适应分配算法                                                                                                                                                                                                  | 
| Differentially Private Federated Learning on Heterogeneous Data                                               | Noble et al.  | AISTATS/2022 | 基于Non-IID，在原有的联邦算法SCAFFOLD的基础上加上DP形成DP-SCAFFOLD算法                                                                                                                                                                                                      | 
| User-Level Privacy-Preserving Federated Learning: Analysis and Performance Optimization                       | Wei et al.  | TMC /2021 | 在客户端采样而不是客户端中的样本采样，但同样套用了MA的隐私度量方式（合理性存疑），后面加了一个基于隐私预算自适应最优迭代次数的方法，其实就是变相的自适应隐私预算分配。本篇论文是逐样本裁剪，且对于相邻数据集敏感度的选择看起来是simple-level[【vedio】](https://www.bilibili.com/video/BV16A4y1X74k?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3) |
| Adap DP-FL: Differentially Private Federated Learning with Adaptive Noise                                     | Fu et al. | Trustcom/2022 | 主要是两个自适应点，一个是不同客户端不同轮的自适应裁剪范数，一个是不同轮的自适应噪声系数衰减。                                                                                                                                                                                                        | 
| Adaptive Local Steps Federated Learning with Differential Privacy Driven by Convergence Analysis              | Ling et al. | arxiv/2023 | 利用收敛性分析，在有限的隐私预算和通信成本下求出最优的本地迭代轮数。                                                                                                                                                                                                        | 
| Projected federated averaging with heterogeneous differential privacy                                         | Liu et al.  | VLDB/2021 | 考虑异构隐私预算的场景，并利用客户端提交的具有较高隐私预算的模型更新的顶级奇异子空间，将它们投影到具有较低隐私预算的客户端的模型更新上。                                                                                                                                                                                                        | 
| Federated f-Differential Privacy                                                                              | Zheng et al. | AISTATS/2021 | 样本级的差分隐私保护，提出了强联邦和弱联邦（个人认为弱联邦没什么意义），隐私度量采用GDP                                                                                                                                                                                                          | 
| Practical Differentially Private and Byzantine-resilient Federated Learning                                   | Xiang et al. | PACMMOD/2023 | 设计一个聚合方法来防止拜占庭攻击                                                                                                                                                                                                         | 
| Federated Learning with Differential Privacy: Algorithms and Performance Analysis                             | Wei et al. |TIFS/2019 | 先进行上行链路的隐私保护，然后再基于上行链路的加噪再对下行链路进行加噪，给出了收敛性分析。相邻数据集的定义是sample-level的| 
| SoteriaFL: A unified framework for private federated learning with communication compression                  | Li et al. |NIPS/2022 | 进行模型压缩和量化·，来降低DP-HFL的通讯成本| 
| Private, Efficient, and Accurate: Protecting Models Trained by Multi-party Learning with Differential Privacy | Fudan University | SP/2022 | 核心在于利用了多方安全的秘密分享构建出一个虚拟的联邦中心方，使的不同客户端的样本数据可以进行秘密分享后“集中”式训练，然后在再训练的梯度上加DP，以此满足模型的差分隐私。相较于之前的模型，该模型不用LDP,也不用shuffle，将所以客户端数据整合成集中式变成CDP的形式（这样可以用更小的eps，即不造成更大的精度损失），并且没有可信第三方。| 
| DP-ADMM: ADMM-based distributed learning with differential privacy                                            | Huang et al. | TIFS/2019 | 用ADMM结合DP进行本地梯度下降，提出DP-ADMM| 
| Differentially private federated learning via inexact ADMM with multiple local updates                        | Ryu et al. | arxiv/2022 | 本地进行多轮的DP-ADMM。| 
| DPAUC: Differentially Private AUC Computation in Federated Learning                                           | Jiankai Sun | 2022 | 作者举了一个联邦下客户端模型根据本地测试集进行性能评测算出AUC给中心方的场景，该场景认为AUC（ROC curve）会泄露客户端本地测试集隐私信息，故在传给中心方相关FP和FN等数值上加拉普拉斯噪声进行隐私保护。                                                                                                                                            | 
| Federated learning with bayesian differential privacy                                                         | Triastcyn et al.  | Big data/2019 | 利用联邦学习中各个客户端数据分布相似从而更新分布也相似的假设，将BDP引入差分隐私联邦学习                                                                                                                                           | 

##### CL-DP
保护的是客户端的参与信息,将客户端的模型看成一个数据,联邦中心方一般是可信的。
这里的保护对象就是client，保护的是中心方聚合后的模型，多一个或少一个client上传模型，不会明显的影响这个聚合后的模型。这个分类下，有些是本地加一部分噪声然后到中心满足DP的这种，可以看成先满足少量保护的LDP（对于敌手是服务器而言），然后再满足一个eps的DP。

| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| Learning differentially private recurrent language models | Mamahan et al. |  Learning/2017 |首次提出了DP-FedAvg和DP-FedSGD，采样是对客户端进行采样，加噪在中心方。敏感度的计算基于采样率，每个client的联邦权重得到。[【vedio】](https://www.bilibili.com/video/BV1fd4y1A7LD?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)| 
|Differentially private federated learning: A client level perspective| Geyer et al. | Cryptography and Security/2017 | 和DP-FedAvg的区别在于客户端上传的模型在中心方进行裁剪，这样就可以进行自适应的裁剪操作（比如取各个客户端模型的范数中值作为裁剪范数）|
| Differentially private learning with adaptive clipping | Andrew et al. | NIPS/2021  | 自适应裁剪，用模型参数范数分布的分位数来作为裁剪范数   |                                                                                                                                                                                     
| Understanding clipping for federated learning: Convergence and client-level differential privacy | Zhang et al. | ICML/2022  | 比较模型参数裁剪和梯度裁剪，用收敛性证明了梯度裁剪方法比参数裁剪更好    |                                                                                                                                                                                    
| Differentially Private Federated Learning with Local Regularization and Sparsification | Cheng et al.  |CVPR/2022                                               | 本地对模型之差进行裁剪加噪，并提出了对应的正则化和稀疏化方法                                                                                                                                                                                        
| Make Landscape Flatter in Differentially Private Federated Learning | Shi et al.                | CVPR/2023                                             | 提出的算法集成了Sharpness Aware Minimization（SAM）优化器，以生成具有更好稳定性和权重扰动鲁棒性的局部平面度模型，这导致局部更新的范数较小                                                  
| Personalization improves privacy-accuracy tradeoffs in federated learning | Bietti et al.                 | ICML/2022                                             | 利用本地个性化的模型来增强全局模型性能                                                 
| Learning to generate image embeddings with user-level differential privacy | Xu et al.                 | CVPR/2023                                            | 提出模型参数 softmax 层的大小与样本标签的数量呈线性关系。通过将softmax层保持在本地而不上传，则更多的客户端（即更多的样本标签）参与联合学习不会导致更多的噪声注入。                                                
| Dynamic personalized federated learning with adaptive differential privacy | Yang et al.                 | NIPS/2023                                            | 结合layer-wise Fisher information，在本地动态保存高信息模型参数，使其免受噪声影响。他们还引入了一种自适应正则化策略，对上载到服务器的模型参数施加差分约束，增强了对剪切的鲁棒性。                                                

###### CL with SA
在联邦中心方不可信的情况下提出用安全聚合实现CL-DP。一般的做法就是本地添加离散的噪声满足一定DP，然后加密，然后中心方聚合加密的结果，这个结果解密后是满足CL-DP的。

| Title | Team/Main Author | Venue and Year | Key Description                                                                                         
| :------------| :------ | :---------- |:-------------------------------------------------------------------------------------------------------- 
| cpSGD: Communication-efficient and differentially-private distributed SGD | Agarwal et al.|  NIPS/2018 | 提出伯努利机制进行离散加噪，满足DP                                                                                      | 
|The poisson binomial mechanism for unbiased federated learning with secure aggregation| Chen et al.  | ICML/2022 | 用泊松伯努利机制将高斯噪声离散化                                 |
| Compression boosts differentially private federated learning | Kerkouche et al.  | SP/2021  | 自模型稀疏化来降低通信成本                                                                             |                                                                                                                                                                                     
| Efficient differentially private secure aggregation for federated learning via hardness of learning with errors | Stevens et al.  | USENIX/2022  | 该文章在原来的用同态加密进行联邦聚合的基础上，刻画了噪声的DP衡量，因为之前用的LWE天然存在噪声，所以该文章把这个噪声用DP量化出来。同时，也可以理解成其用同态加密的方法将原来的LDP加噪变成CDP的加噪方式，类似shuffle的理念                                                                     |                                                                                                                                                                                    
| D2P-Fed: Differentially private federated learning with efficient communication | Wang et al.   |arxiv/2020                                             | 用离散高斯机制将高斯噪声离散化                                                                        
| The distributed discrete gaussian mechanism for federated learning with secure aggregation | Kairouz et al.              | ICML/2022                                            | 扩展离散高斯，提出恶意客户端比例相关的新的隐私分析                    
| The fundamental price of secure aggregation in differentially private federated learning | Chen et al.                  | ICML/2022                                             | 离散高斯的基础上，通过随机向量投影降低通讯成本                                                                                    
| The skellam mechanism for differentially private federated learning | Agarwal et al.               | NIPS/2021                                          | 提出Skellam机制来将高斯噪声离散化               


#### LDP-HFL
上面的会把模型或者模型之差看成一个整体，将其看成一个高维数据。
LDP-HFL 可视为基于 LDP 的均值估计问题，因为每个模型参数都代表一个高维连续数据点。每个客户端在本地扰动其模型参数，然后发送给服务器，服务器汇总数据并生成均值估计结果。

| Title | Team/Main Author             | Venue and Year | Key Description                                                                                                                                                                                                                                                                               
| :------------|:-----------------------------|:---------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| Collecting and analyzing multidimensional data with local differential privacy  | JWang et al.                 | ICDE/2019      | 提出PM和HM                                                                                                                                                                                                                                                                                       | 
| Fedsel: Federated sgd under local differential privacy with top-k dimension selection | Liu et al.                   | DASFAA/2020    | 两阶段LDP-FL框架，先进行TOP-K参数选择，再用PM机制进行扰动                                                                                                             | 
| PPeFL: Privacy-Preserving Edge Federated Learning with Local Differential Privacy | Wang et al.                  | IOT/2023       | 提出DMP-UE 机制来扰动 Top-$K$ 参数以获得高效用，该机制在 Duchi~cite{duchi2013local} 的基础上进行了扩展，输出三种情况（包括输出 0），而不是两种情况。他们还通过引入边缘节点来执行边缘聚合，从而降低了通信成本。                                                                                                                                                                                              
| Local differential privacy-based federated learning for internet of things | Zhao et al.                  | IOT/2020       | 基于Duchi提出Three-output机制，拥有三个输出可能，包括0                                                                                                                                                                                                      
| Federated learning with personalized local differential privacy | Yang et al.                  | ICCCS/2021          | 各个客户端不同的隐私需求
| Federated latent dirichlet allocation: A local differential privacy based framework | Wang et al.                  | AAAI/2020          | FedLDA,提出先验的随机相应机制结合LDA模型                                                                                                                                                                                                
| Federated latent dirichlet allocation: A local differential privacy based framework | Zhang et al.                 | Software/2023         | 基于伯明汉距离提出个性化联邦学习                                                                                                                                                                                               
| Webfed: Cross-platform federated learning framework based on web browser with local differential privacy | Lian et al.                  | ICC/2022        | 跨浏览器的LDP联邦学习                                                                                                                                                                                               
| Local differential privacy for federated learning | Mahawaga et al.              | ESORICS/2022       | 首先使用 CNN 模型中的卷积层和池化层从本地数据集中提取特征向量。然后，将这些特征向量转换为扁平化的一维向量，并通过 RAPPOR 进行一元编码和扰动，最后上传到服务器。                                                                                                                                                                                               
| Safeguarding cross-silo federated learning with local differential privacy | Wang et al.                  | DCN/2022         | 用PAPPOR进行本地扰动，并提出可以抵御重构攻击                                                                                                                                                                                             
| Signds-fl: Local differentially private federated learning with sign-based dimension selection | Jiang et al.                 | TIST/2022       | 将TOP-K扩展到多维参数选择，并提出多维下的EM机制                                                                                                                                                                                              
| FedTA: Locally-Differential Federated Learning with Top-k Mechanism and Adam Optimization | Li et al.                    | ICUS/2022        | 在TOP-K的基础上提出本地进行Adam下降，然后添加拉普拉斯噪声
| LDP-FL: Practical Private Aggregation in Federated Learning with Local Differential Privacy | Sun et al.                   | IJCAI/2021        | 提出参数shuffle，认为参数shuffle后不需要对维度进行隐私预算分割


#### Shuffle model-HFL
  
shuffle相关的联邦文章本质上从把隐私保证LDP转到CDP，shuffle 后最终保护的概念也是让中心方无法区分哪个client上传了数据，所以是client level。但是由于shuffle目前技术比较成熟，而且文章数量多，我们可以单独拿出来讨论，这种应该算是LDP和CDP的一种技术混合。

##### Shuffle model with SL-DP
| Title                                                                                                   | Team/Main Author            | Venue and Year                                     | Key Description                                                                                                                                                                                                                                                                               
|:--------------------------------------------------------------------------------------------------------|:----------------------------|:---------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| FLAME: Differentially Private Federated Learning in the Shuffle Model                                   | Liu et al. | AAAI/2021                                          | 这个是标准LDP结合联邦，裁剪是对每个神经元进行裁剪，隐私预算是分配到每个神经元上的。提出了SS-Double协议，即通过随机采样的方式加shuffle(文章需要idx做标记，应该是对维度进行shuffle)双重隐私放大，通过数值填充的方式将采样提供的的隐私放大效应与混洗模型的隐私放大效应进行组合。扰动方式是对梯度编码然后随机扰动。整体框架基于ESA。                                                                                                             | 
| A Generalized Shuffle Framework for Privacy Amplification: Strengthening Privacy Guarantees and Enhancing Utility                                      | Chen et al.                     | arxiv/2023                                      | 对客户端进行采样，然后再对用户的样本进行采样。以样本作为单位                                                                                                                                                                                              

##### Shuffle model with CL-DP
| Title                                                                                                   | Team/Main Author            | Venue and Year                                     | Key Description                                                                                                                                                                                                                                                                               
|:--------------------------------------------------------------------------------------------------------|:----------------------------|:---------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| Shuffled check-in: privacy amplification towards practical distributed learning                                  | Liew et al. | arxiv/2023                                       | 提出洗牌+自采样，及将洗牌模型和泊松采样结合                                                                                                            | 
| FLAME: Differentially Private Federated Learning in the Shuffle Model                                   | Liu et al. | AAAI/2021                                          | 这个是标准LDP结合联邦，裁剪是对每个神经元进行裁剪，隐私预算是分配到每个神经元上的。提出了SS-Double协议，即通过随机采样的方式加shuffle(文章需要idx做标记，应该是对维度进行shuffle)双重隐私放大，通过数值填充的方式将采样提供的的隐私放大效应与混洗模型的隐私放大效应进行组合。扰动方式是对梯度编码然后随机扰动。整体框架基于ESA。                                                                                                             | 
| Echo of Neighbors: Privacy Amplification for Personalized Private Federated Learning with Shuffle Model | Liew et al.                | arxiv/2023                                              | 本提出洗牌+自采样，及将洗牌模型和泊松采样结合                                                                                                                                                                                                     

### Incentive
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| Pain-FL: Personalized Privacy-Preserving Incentive for Federated Learning|  Chinese University of Hong Kong | IEEE Journal on Selected Areas in Communications /2021 | 基于差分隐私联邦学习的激励，tradeoff点在于客户端希望多加噪声保护本地数据，中心方希望客户端少加噪声获得高准确率模型，于是一开始就签订协议，协议对应的是噪声保护程度和奖励，即越少加噪，就获得奖励越多。但是噪声保护程度各个客户端一开始就定好了，假定各个客户端是理性的。中心方根据签订好的锲约，进行客户端采样联邦，选择隐私保护力度小的客户端参与联邦。|

### VFL
这个是未来可以研究的一个方向，针对纵向联邦的分层网络的场景进行差分隐私保护

### TFL
[视频讲解](https://www.bilibili.com/video/BV14g411V7nZ?p=8&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)

## Attack
[联邦学习中的攻击与防御](https://www.bilibili.com/video/BV14g411V7nZ?p=2&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)

[视觉算法的模型攻防](https://www.bilibili.com/video/BV1C34y1n7vV?spm_id_from=333.999.0.0)

| Title                                                                                   | Team/Main Author           | Venue and Year                                  | Key Description                                                                                                                                                              
|:----------------------------------------------------------------------------------------|:---------------------------|:------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| Deep leakage from gradient                                                              | Renmin University of China | Neurips /2019                                   | 先用虚拟的数据和标签得到虚拟模型梯度，计算虚拟模型梯度和真正的梯度之间的L2损失，损失求导使得虚拟梯度拟合真实梯度，从而使得虚拟数据和虚拟标签同步变化，最后复现出原始数据。该方案仅适用于单层网络模型。                                                                         | 
| Towards Security Threats of Deep Learning Systems: A Survey                             | Yingzhe He                 | IEEE Transactions on Software Engineering /2021 | 深度学习下的攻击综述，包括模型窃取攻击，模型反演攻击，投毒攻击，对抗性样本攻击                                                                                                                                      |                                                                                               | 
| Local and Central Differential Privacy for Robustness and Privacy in Federated Learning | Openmind                   | NDSS /2021                                      | 联邦场景下LDP（其实是联邦下的样本级差分隐私）和CDP（联邦下的客户端级差分隐私）对推理攻击（隐私性）和后门攻击（鲁棒性）的防御研究，内有大量实验设置和对比结果。DP之所以能防御后门攻击可能在于加DP的同时同步给置入后门的攻击方加了噪声，削弱了攻击方能力。                                            |                                                                                               | 
| Comprehensive Privacy Analysis of Deep Learning | University of Massachusetts Amherst                   | NDSS /2018                                      | 在联邦场景下假设联邦中心方或其他参与方是敌手的情况下的成员推理攻击，并分了无监督，半监督，白盒，黑盒，积极攻击，消极攻击等多种场景。主要在于它的成员推理攻击比较普适，攻击的模型思想比较清晰。                                                                              |                                                                                               | 
| Evaluating Differentially Private Machine Learning in Practice       | openmind            | CCS /2019                                       | 对不同的DP（高级组合，ZCDP，RDP等）进行评估，用黑盒和白盒的成员推理攻击来评估经过这些DP训练后的模型的防御效果。                                                                                                                |                                                                                              
| FLAME: Taming Backdoors in Federated Learning       | Technical University of Darmstadt            | USENIX /2022                                    | 在差分隐私结合联邦场景中提出了FLAME， 使用模型聚类（基础假设为后门训练的权重和良性权重存在较大的方向偏差）和权重裁剪（采用中位数进行裁剪，基础假设是后门敌手不会超过总客户端的一半）方法。这确保了 FLAME 可以在有效消除对抗性后门的同时保持聚合模型的良性性能。实验表明加很少的噪声就可以非常有效的抵御后门攻击，应该是裁剪对后门的作用大。 |                                                                                              

### Membership Inference Attack
| Title                                                                                   | Team/Main Author           | Venue and Year                                  | Key Description                                                                                                                                                              
|:----------------------------------------------------------------------------------------|:---------------------------|:------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| Gradient-Leaks: Enabling Black-box Membership Inference Attacks against Machine Learning Models       | HUST            | TIFS /2023                                    | 因为每条数据都会对梯度更新产生影响，利用每个数据独有的梯度，用自编码器得出梯度背后的隶属度信息（瓶颈为隶属度相关的几个特征），无监督学习；利用构造局部模型的方式求某条data的梯度（类似可解释性AI中的LIME方法） |                                                                                              
| TEAR: Exploring Temporal Evolution of Adversarial Robustness for Membership Inference Attacks Against Federated Learning       | HUST            | TIFS /2023                                    | 每个client能访问本地的全部训练信息，并在每一轮全局下发的时候得到全模型预测接口，可进行label only的预测；攻击方client用自身数据的对抗鲁棒性作为特征，训练二分类模型进行对其他client的MIA（本文假设各个client数据异质性不强）。攻击时利用对抗鲁棒性随时间变化（收敛趋势）的特性得出隶属信息；利用fL决策边界会逐渐拟合训练集而非测试集；决策边界和训练数据的额距离越大，越不易受对抗性扰动，将此距离作为对抗鲁棒性的度量，训练过程中持续收集该性质，作为隶属度特征。 |                                                                                              
| Membership Inference via Backdooring       | Auckland            | IJCAI /2022                                    | 这个场景是数据拥有者判断自己的敏感数据集有没有用于某个模型训练，对自己数据集的少部分进行标记（trigger），利用这部分数据进行隶属推断，模型将学到trigger和目标标签的相关性。利用成员推理来判断自己标记的数据集在不在训练数据集中 |                                                                                              


### copyright protection
一般使用后门植入进行版权保护 

| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :-----------------------
| Radioactive Data: Tracing Through Training       | Facebook            | PMLR /2020                                    | 对于数据集版权保护问题，通过在样本中引入不可察觉的变化（放射性），来跟踪特定的数据是否被用于训练模型，根据模型输出判断被保护的数据是否被用于该模型的训练。在数据的潜在空间中添加标记（一个方向向量），使用标记将特征移向某个方向，再把标记从特征空间反向传回像素；这些标记在整个训练过程中保持可检测，最后算分类层权重和这个标记向量的余弦相似度，通过统计检验p-value来验证是否使用了标记数据 |                                                                                              

## Application Scenarios

### Text Protection
[结合差分隐私的文本保护](https://www.bilibili.com/video/BV1it4y147vD?spm_id_from=333.999.0.0)   

### Recommended System
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| Differentially Private User-based Collaborative Filtering Recommendation Based on K-means Clustering | Zhili Chen | Expert Systems With Applications /2018 | 协同过滤添加差分隐私，先进行聚类，聚类后对待预测的用户所在的簇（簇中用户数m）选n个相似用户，根据皮尔逊距离用排列组合组合出$C_m^n$个组合，然后用差分隐私的指数机制来选择最近的那个组合。本质时是用差分隐私的指数机制来选择和被预测用户最相关的那个组合用户| 
| Differentially private matrix factorization | Nanjing University | AAAI /2015 | 基于矩阵分解的推荐系统，本质上是将一个权重参数分成一个用户矩阵和一个物品矩阵，两个矩阵都可以看成待训练的权重。然后根据原本的一个大矩阵中已经打分的数值进行权重训练，最后得到这两个权重矩阵。文章给出了本地差分隐私的保护，即把每个用户自己的权重进行加噪再上传到中心方联邦平均。| 

### DP and image
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| DPGEN: Differentially Private Generative Energy-Guided Network for Natural Image Synthesis | Jia-Wei Chen | CVPR /2022 | 该文章基本基于《Generative Modeling by Estimating Gradients of the Data Distribution》的基础上添加差分隐私，就是在学习已有图片信息然后生成新的图片信息的过程中加差分隐私。存在两点问题，一是在做指数机制时其从当前数据中再取K个进行随机输出，这边K的取值没有进行隐私保护，二是其方法没有生成样本对于的标签信息，作者认为用生成的满足差分隐私的图像数据去查询标签不会泄露隐私，但其实会有隐私问题[【vedio】](https://www.bilibili.com/video/BV1me411V7pV/?spm_id_from=333.999.list.card_archive.click&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)| 



### DP and cytro
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| Efficient Differentially Private Secure Aggregation for Federated Learning via Hardness of Learning with Errors | University of Vermont | CCS/2021 | 该文章在原来的用同态加密进行联邦聚合的基础上，刻画了噪声的DP衡量，因为之前用的LWE天然存在噪声，所以该文章把这个噪声用DP量化出来。同时，也可以理解成其用同态加密的方法将原来的LDP加噪变成CDP的加噪方式，类似shuffle的理念。[【vedio】](https://www.bilibili.com/video/BV1fR4y1D7dU/?spm_id_from=333.999.list.card_archive.click&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)| 
| Private, Efficient, and Accurate: Protecting Models Trained by Multi-party Learning with Differential Privacy | Fudan University | SP/2022 | 核心在于利用了多方安全的秘密分享构建出一个虚拟的联邦中心方，使的不同客户端的样本数据可以进行秘密分享后“集中”式训练，然后在再训练的梯度上加DP，以此满足模型的差分隐私。相较于之前的模型，该模型不用LDP,也不用shuffle，将所以客户端数据整合成集中式变成CDP的形式（这样可以用更小的eps，即不造成更大的精度损失），并且没有可信第三方。| 


### DP and auditing
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| Privacy Auditing with One (1) Training Run | Thomas Steinke | arxiv/2023 | 该文章提出的DP隐私审计更像是一种攻击，通过在训练集中随机插入数据，然后再对宣传满足DP的算法进行训练，通过观测输出来判断出入的数据有没有在训练数据集中，然后得到对应的eps。一般来说，文章认为当前的DPSGD算法的理论隐私下界太高，这篇文章通过经验实验给的隐私上界一般更紧。| 


## Meachine unlearning
## Unlearning in Centralized machine learning 
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :-----------------------
| When Machine Unlearning Jeopardizes Privacy | CMU | CCS/2021 | 提出unlearn会造成隐私泄露，可以根据unlearn前后单目标的后验概率的差异判断数据是否被删；两种聚合unlearn前后的后验概率的方案，作为攻击模型的输入，concatenating（在过拟合模型攻击效果好）/求差（在泛化性能好的模型攻击效果好）。2个指标衡量损失了多少隐私（目标样本置信度高于经典MIA的比例&置信度增量的平均值）并提出4个防御方案| 

## Unlearning in FL
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :-----------------------
| FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models | HUST | IWQOS/2022 | 牺牲中心方的内存，每隔几轮存储各个client的更新增量；指出client的更新指示全局模型变化的方向；利用少量的矫正训练的到的更新指示方向，用保存的更新提供幅度，得出unlearn后的更新（旧update的幅值+矫正训练update的方向）；用类似相位的方法评价unlearn后的模型与retrain的模型的相似度| 
| Federated Unlearning: How to Efficiently Erase a Client in FL?  | IBM | Arxiv/2022 | client自行完成unlearn，无需中心方全局访问数据，无需存储历史记录；client对自身训练过程有较大权限。请求遗忘的client自行用下发的全局模型，和前一时刻自身局部模型，算出其他client的平均（近似）作为参考模型；做用PGD梯度上升（PGD），约束为和参考模型的距离不能太大| 
| Federated Unlearning via Class-Discriminative Pruning  | PolyU | WWW/2022 | 专注图像分类任务，忘记一类标签数据，每个client都需上传通道和类别之间的关联度，中心方聚合后，用TF-IDF指标评估相关性，剪枝，下发，微调恢复精度。client上传的关联度类似可解释性AI中的可视化方法；TF- IDF时NLP中衡量word与一堆文档中某个文档的关联性的指标；本文算法不需明确具体需要删除的data，删的是一类label| 
| FedRecovery: Differentially Private Machine Unlearning for Federated Learning Frameworks  | UTS | TIFS/2023 | 引入梯度残差的概念来量化增量效应，全局模型中删除梯度残差的加权和来消除某个客户端的影响，并添加特定的高斯噪声，使得unlearn模型和retrain模型在统计上不可区分。梯度残差通过计算前一时间点的模型与当前模型的梯度差得到。| 

