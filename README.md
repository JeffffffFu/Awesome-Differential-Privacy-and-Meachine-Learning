# Papers

<!-- [![Stars](https://img.shields.io/github/stars/youngfish42/Awesome-Federated-Learning-on-Graph-and-Tabular-Data.svg?color=orange)](https://github.com/youngfish42/Awesome-Federated-Learning-on-Graph-and-Tabular-Data/stargazers)  [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re) [![License](https://img.shields.io/github/license/youngfish42/Awesome-Federated-Learning-on-Graph-and-Tabular-Data.svg?color=green)](https://github.com/youngfish42/image-registration-resources/blob/master/LICENSE) ![](https://img.shields.io/github/last-commit/youngfish42/Awesome-Federated-Learning-on-Graph-and-Tabular-Data) -->

bilibili（论文分享）:https://space.bilibili.com/80356866/dynamic

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
  - [DP and Meachine Learning](#dp-and-meachine-learning)  
    - [Meachine Leaning](#meachine-learning)
    - [Meachine Leaning with DP](#meachine-learning-with-dp)
  - [DP and Federated Leaning](#dp-and-federated-learning)  
    - [Federated Leaning](#federated-leaning) 
    - [Horizontal FL with DP](#horizontal-fl-with-dp)
      - [client-level](#client-level)
      - [samping-level](#samping-level)
      - [LDP FL](#ldp-fl)
    - [vertical FL with DP](#vertical-fl-with-dp)
    - [incentive](#incentive)
  - [Attack](#attack)
  - [Application Scenarios](#application-scenarios)
    - [Text Protection](#text-protection)
    - [Recommended System](#recommended-system)
    - [DP and image](#dp-and-image)


## DP Theory

### Differential Adversary Definition
CDP（central DP）有一个完全可信的中心方，敌手是外界。而LDP（local DP）认为中心方是诚实但好奇的。
#### CDP
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| Differential privacy | Cynthia Dwork | ICALP/2006 | 首次提出差分隐私的定义 | 
| Programming Differential Privacy （Book）| Joseph P. Near and Chiké Abuah | 2021 | 讲诉了DP的概念定理和机制等，并附有简单代码呈现（简单入门推荐） | 
| The Algorithmic Foundations of Differential Privacy（Book） | Cynthia Dwork | 2014 | DP的定义理论，高级组合和相关机制等的完整证明推导（更加理论）[【拉普拉斯、严格差分、高斯机制、松弛差分】](https://www.bilibili.com/video/BV18r4y1j7Bs?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3) |
 | Differential Privacy From Theory to Practice （Book）| Ninghui Li | 2014 | 除了一些基本定理和机制，用了具体的实际例子讲诉了DP的用法及DP的伦理探讨（更加实用化）[【Chapter1、Chapter2】](https://www.bilibili.com/video/BV1br4y1J7Qn?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3),[原作者讲解]()|

#### LDP
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| 本地化差分隐私研究综述 | 中国人民大学 | Journal of Software/2018 | 介绍了本地化差分隐私的原理与特性,总结和归纳了LDP当前研究工作,重点阐述了频数统计、均值统计的LDP机制设计[【vedio】](https://www.bilibili.com/video/BV18B4y1a75b?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3) | 
| RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response | Google | ACM SIGSAC /2014 | 1.RAPPOR的分类和对应扰动方法。2.Differential Privacy loss of RAPPOR（主要是多维） | 
| Locally Differentially Private Protocols for Frequency Estimation | Purdue University | USENIX/2017 | 1.提出了一个Pure LDP Protocols，并基于Pure LDP Protocols给出了其方差和频数估计的公式。2.提出OUE（UE=basic RAPPOR），给出了q的最佳选择和方差。[【vedio】](https://www.bilibili.com/video/BV1HW4y127mp?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3) |  |

### Privacy Measurement Method

#### DP
其中分为严格的差分隐私和松弛的差分隐私，DP隐私损失度量可以查阅上面的CDP相关文献。

#### RDP(MA)
此前的MA（monments accountant）目前来看就是RDP，区别在于RDP从一开始的散度就用了高阶矩进行度量，而MA是在分布上进行高阶矩度量，不过最后的表达形式几乎一致，两者思想也一致。
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| Deep Learning with Differential Privacy | Martín Abadi |  CCS/2016 | 首次提出高斯下采样机制并利用Moments accountant技术进行隐私损失度量 [【vedio】](https://www.bilibili.com/video/BV1r44y1u7iP?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)| 
| Rényi Differential Privacy | Ilya Mironov |  Security Foundations Symposium/2017 | Rényi 差分隐私的定义、性质和结论。主要关注里面的高斯机制（无下采样），组合性质及转为DP的公式。[【vedio】](https://www.bilibili.com/video/BV1YF411L7xV?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3) | 
| Subsampled Rényi Differential Privacy and Analytical Moments Accountant | Yu-Xiang Wang | AISTATS /2018 | RDP和MA的直观对比，主要给出了一个RDP高斯下采样的一个隐私损失上界，有很多论文引用这个上界的结论。 | 
| Rényi Differential Privacy of the Sampled Gaussian Mechanism | Ilya Mironov | 2019 | 给出了Rényi Differential Privacy of the Sampled Gaussian Mechanism 各种情况的隐私损失公式，里面的3.3的公式目前应用于各个开源库的隐私计算中（更为简洁） |
| Hypothesis Testing Interpretations and Rényi Differential Privacy | Balle B | AISTATS/2020 | 定理21给出了更为紧凑的RDP转DP的公式（目前开源库opacus应用的是这个转换） |

#### ZCDP
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds | UMark Bun | 2016 | 提出ZCDP的定义及ZCDP和DP的关系（ZCDP也是从瑞丽散度来定义的）。不过和RDP不同的是，ZCDP的$\alpha\in(1,+\infty)$，如果细致的选择$\alpha$，RDP的计算出的隐私损失要小于ZCDP。[RDP与ZCDP对比参考](https://zhuanlan.zhihu.com/p/457972991)| 


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


## DP and Meachine Learning
本章集中在如何进行更高效机器学习集合差分隐私训练
### Meachine Learning
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation | Sebastian Bach | PLOS ONE/2015| 利用LRP（分层相关传播）算法，反向求出每个神经网络中每个元素的相关性贡献和输入像素对预测函数f的贡献| 
### Meachine Learning with DP
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| Deep Learning with Differential Privacy | Martín Abadi | CCS/2016| 首次将差分隐私和深度学习梯度下降结合提出DPSGD算法 [【vedio】](https://www.bilibili.com/video/BV1r44y1u7iP?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3) | 
| Differentially Private Model Publishing for Deep Learning | Gatech | SP/2019| 1.提出了ZCDP的下采样机制。2.提出了自适应的噪声系数下降 | 
| Concentrated Differentially Private Gradient Descent with Adaptive per-Iteration Privacy Budget | University of Georgia | ACM SIGKDD/2018| 基于梯度的变化来自适应进行隐私预算分配，其实就是根据梯度的变化来进行噪声系数的自适应变化，思想是梯度大的时候多加噪，梯度小的时候少加噪 | 
| An Adaptive and Fast Convergent Approach to Differentially Private Deep Learning| University of Nanjing | INFOCOM/2020 | 1.在原有的梯度下降法RMSPROP上加上DP。2.提出了梯度中逐元素的自适应裁剪，这个裁剪值根据RMSPROP方法中每轮每个元素积累的值来衡量 [【vedio】](https://www.bilibili.com/video/BV1qr4y1M7sq?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)| 
| Tempered Sigmoid Activations for Deep Learning with Differential Privacy| Nicolas Papernot | AAAI/2021 | 在原有的DPSGD上，对于神经网络结构中的激活函数用缓和的sigmod函数-tanh（有界函数）来替代Relu（无界函数），有界函数可以使得梯度控制在某个阈值中，从而让裁剪范数C不会大于梯度范数值，fashionmnist和cifar10按照它的超参数无法达到对应论文的准确率| 
| Semi-Supervied Knowledge Transfer For Deep Learning From Private Training Data| Nicolas Papernot  | ICLR/2017 | 提出了差分隐私结合机器学习的新范式，构建teacher model（私有数据集）的标签投票作为输出的DP query，利用标签也就不可避免得引入了knowledge transfer的方式来将student model（公共数据集）作为原模型的影子[【vedio】](https://www.bilibili.com/video/BV1W34y1775e?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)| 
| Three Tools for Practical Differential Privacy | Koen Lennart van der Veen   | Machine Learning/2018| 提出三种提高DPSGD精度的方法：1、训练前进行完整性检查。2、进行分层裁剪（没给隐私度量方式）3、大批量训练（每次训练采样率加大）| 
| Differentially Private Learning Needs Better Features (or Much More Data) | Stanford | ICLR/2021| 该论文以Nicolas的tanh优化为baseline进行比对，提出了人工的特征提取（handcrafted features）的方式，用了ScatterNet先进行特征提取，分别用ScatterNet+Linear model和ScatterNet+CNN都有很好的效果，目前看起来是最高的准确率（但是添加了预处理的步骤），该论文有代码，提供了相关超参数| 
| Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger | AWS AI | ICLR/2022| 该论文以Nicolas的tanh优化为baseline进行比对，提出了最小化裁剪范数，然后整个裁剪加噪后的式子可以简化成裁剪范数和学习率的线性关系。作者提出只要取一个很小的裁剪范数，然后调整学习率就可以获得高的准确率，即固定一个很小的裁剪范数即可（范数裁剪不会影响梯度方向，因为是整体线性裁剪）。后面作者加了一个gamma在裁剪的时候使得收敛性得以证明，也比之前的baseline效果好一点点。这篇文章的结合DP的收敛性分析值得借鉴。| 
| DPNAS: Neural Architecture Search for Deep Learning with Differential Privacy | Chinese Academy of Sciences | national conference on artificial intelligence/2021| 这篇论文是寻找最适合DP深度学习的网络结构，文章会在每层都提供一些网络设计，然后遍历的去尝试，哪种组合会更好，这篇文章实现的准确率比上一篇还高。他们发现了SELU激活函数表现比tanh好。| 
| Label differential privacy via clustering | arXiv|Google/2021|提出了标签差分隐私，即对标签进行KRR扰动满足差分，但是在对标签扰动之前先进行聚类，使得相同分布的样本标签在同一簇中，聚类的过程有进行差分隐私加噪在标签上。然后簇中进行样本标签的扰动。最后在loss函数中行乘以对应扰动机制的逆进行矫正。以此完成图像分类任务。 | 
## DP and Federated Learning

### Federated Leaning
想看更多的联邦学习文献推荐可以转到 :

[Awesome-Federated-Learning-on-Graph-and-GNN-papers](https://github.com/huweibo/Awesome-Federated-Learning-on-Graph-and-GNN-papers) 

 [Awesome-Federated-Machine-Learning](https://github.com/innovation-cat/Awesome-Federated-Machine-Learning#16-graph-neural-networks)

 [个性化联邦学习代码仓](https://github.com/microsoft/PersonalizedFL)

| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| A Survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection | NUS | IEEE Transactions on Knowledge and Data Engineering/2021| 从各个维度对联邦学习进行总结分类，还有联邦学习主流开源库的对比，是比较完善全面的联邦学习综述 | 
| Communication-efficient learning of deep networks from decentralized data | H Brendan McMahan | AISTATS/2017| 首次提出了FedAvg算法，即本地迭代多轮再参与联邦平均，可以大大的节省通讯成本 | 
| Federated Optimization in Heterogeneous Networks | University of Pennsylvania |  Proceedings of Machine Learning and Systems/2020 | 针对Non-iid场景提出FedProx算法，与FedAvg的区别就是在客户端本地的损失函数上加了正则项（同时考虑中心方模型）防止本地迭代过拟合 | 
| Adaptive Federated Learning in Resource Constrained Edge Computing Systems| Shiqiang Wang | JSAC/2019 | 在本地迭代资源与中心方聚合资源受限的情况下，最优化本地迭代次数，即最优化客户端参与中心方聚合的频率（文章的引言讲到如果不考虑资源受限，本地迭代一次就参与联邦是最好的）| 

### Horizontal-FL with DP

#### client-Level
保护的是客户端的参与信息,将客户端的模型看成一个数据

| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| Learning differentially private recurrent language models | H. B. McMahan |  Learning/2017 |首次提出了DP-FedAvg和DP-FedSGD，采样是对客户端进行采样，加噪在中心方。敏感度的计算基于采样率，每个client的联邦权重得到。[【vedio】](https://www.bilibili.com/video/BV1fd4y1A7LD?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)| 
|Differentially private federated learning: A client level perspective| R. C. Geyer | Cryptography and Security/2017 | 和DP-FedAvg的区别在于客户端上传的模型在中心方进行裁剪，这样就可以进行自适应的裁剪操作（比如取各个客户端模型的范数中值作为裁剪范数）| 
| Federated f-Differential Privacy | University of Pennsylvania | AISTATS/2021 | 样本级的差分隐私保护，提出了强联邦和弱联邦（个人认为弱联邦没什么意义），隐私度量采用GDP| 

#### sample-Level
保护客户端下样本的参与信息，每个客户端下的样本看成一个数据，将联邦中心方看成敌手

| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| Differentially Private Federated Learning on Heterogeneous Data | France Institut Polytechnique de Paris | AISTATS/2022 | 基于Non-IID，在原有的联邦算法SCAFFOLD的基础上加上DP形成DP-SCAFFOLD算法| 
| DP-FL: a novel differentially private federated learning framework for the unbalanced data| Xixi Huang | World Wide Web/2020 | 将DPAGD-CNN算法集合联邦机器学习，DPAGD-CNN是集中式机器学习下差分隐私预算的自适应分配算法| 
| Federated f-Differential Privacy | University of Pennsylvania | AISTATS/2021 | 样本级的差分隐私保护，提出了强联邦和弱联邦（个人认为弱联邦没什么意义），隐私度量采用GDP| 
| User-Level Privacy-Preserving Federated Learning: Analysis and Performance Optimization | Nanjing University of Science and Technology | IEEE Transactions on Mobile Computing /2021 | 在客户端采样而不是客户端中的样本采样，但同样套用了MA的隐私度量方式（合理性存疑），后面加了一个基于隐私预算自适应最优迭代次数的方法，其实就是变相的自适应隐私预算分配。本篇论文是逐样本裁剪，且敏感度的选择看起来是simple-level[【vedio】](https://www.bilibili.com/video/BV16A4y1X74k?spm_id_from=333.999.0.0&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)| 
| Federated Learning with Differential Privacy: Algorithms and Performance Analysis | Nanjing University of Science and Technology | IEEE Transactions on Information Forensics and Security/2019 | 先进行上行链路的隐私保护，然后再基于上行链路的加噪再对下行链路进行加噪，给出了收敛性分析。本篇论文虽然在本地模型加噪，但是敏感度的选择是基于客户端数量，属于client-level| 

| Adap DP-FL: Differentially Private Federated Learning with Adaptive Noise  | Jie Fu | Trustcom/2022 | 主要是两个自适应点，一个是不同客户端不同轮的自适应裁剪范数，一个是不同轮的自适应噪声系数衰减。| 
| DPAUC: Differentially Private AUC Computation in Federated Learning  | Jiankai Sun | 2022 | 作者举了一个联邦下客户端模型根据本地测试集进行性能评测算出AUC给中心方的场景，该场景认为AUC（ROC curve）会泄露客户端本地测试集隐私信息，故在传给中心方相关FP和FN等数值上加拉普拉斯噪声进行隐私保护。| 
#### LDP-FL
没有客户端中心方做小批量梯度下降，一般这种场景一个客户端只有一个样本数据，如果有多个样本数据会训练多个梯度上传给联邦中心方

| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| ESA:A Nove lPrivacy Preserving Framework | Renmin University of China | Journal of Computer Research and Development /2021 | 系统的介绍ESA框架（基于shuffle的LDP）及应用| 
| FLAME: Differentially Private Federated Learning in the Shuffle Model | Renmin University of China | AAAI/2021 | 提出了SS-Double协议，即通过随机采样的方式加shuffle双重隐私放大，通过数值填充的方式将采样提供的的隐私放大效应与混洗模型的隐私放大效应进行组合。扰动方式是对梯度编码然后随机扰动。整体框架基于ESA。| 
| Shuffled Model of Differential Privacy in Federated Learning | Google | AISTATS/2021 | 提出了CLDP-SGD算法，先对客户端进行采样，再对客户端下的样本进行采样，最后对不同客户端的梯度shuffle，这里有多个样本梯度，客户端会传给中心方多个梯度，这边本地基于编码扰动进行加噪。
| Shuffle Gaussian Mechanism for Differential Privacy | Seng Pei Liew | 2022 | 本地客户端进行高斯加噪后得到一个梯度，然后对不同客户端的梯度进行shuffle进行隐私放大。并用不放回随机采样和泊松采样的高斯机制结合shuffle进行理论证明（基于RDP）。
| The Privacy Blanket of the Shuffle Model | Borja Balle  | 2019 | 最为经典的shuffle证明。首先指出，对于shuffle模型，整个shuffle中的数据集应当看成整体，从而才有相邻数据集，针对这个数据集整体满足差分隐私。其次，采用数据相关和数据无关将整体数据分割成两部分，最大化敌手，使得最后只需要分析数据无关的部分，即隐私毯子，随机性在隐私毯子中。[【vedio】](https://www.bilibili.com/video/BV14W4y1b7VK?spm_id_from=333.999.list.card_archive.click&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)
| Hiding Among the Clones: A Simple and Nearly Optimal Analysis of Privacy Amplification by Shuffling | Vitaly Feldman  | 2020 | 和隐私毯子类似的证明思想，不过假设的敌手知道的背景知识没有隐私毯子那么全面，每一个扰动的输出都会影响隐私性。在数据相关部分，进一步分成p/2和p/2进行分析。得到的隐私界比隐私毯子更紧凑。

#### Incentive
| Title | Team/Main Author | Venue and Year | Key Description 
| :------------| :------ | :---------- | :----------------------- 
| Pain-FL: Personalized Privacy-Preserving Incentive for Federated Learning|  Chinese University of Hong Kong | IEEE Journal on Selected Areas in Communications /2021 | 基于差分隐私联邦学习的激励，tradeoff点在于客户端希望多加噪声保护本地数据，中心方希望客户端少加噪声获得高准确率模型，于是一开始就签订协议，协议对应的是噪声保护程度和奖励，即越少加噪，就获得奖励越多。但是噪声保护程度各个客户端一开始就定好了，假定各个客户端是理性的。中心方根据签订好的锲约，进行客户端采样联邦，选择隐私保护力度小的客户端参与联邦。| 

### Vertical-FL with DP
这个是未来可以研究的一个方向，针对纵向联邦的分层网络的场景进行差分隐私保护

[视频讲解](https://www.bilibili.com/video/BV14g411V7nZ?p=8&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)

## Attack
[联邦学习中的攻击与防御](https://www.bilibili.com/video/BV14g411V7nZ?p=2&vd_source=46cfa74ab261e7d7a25c2bfedf5615a3)

[视觉算法的模型攻防](https://www.bilibili.com/video/BV1C34y1n7vV?spm_id_from=333.999.0.0)

| Title                                                                                   | Team/Main Author           | Venue and Year                                  | Key Description                                                                                      
|:----------------------------------------------------------------------------------------|:---------------------------|:------------------------------------------------|:-----------------------------------------------------------------------------------------------------
| Deep leakage from gradient                                                              | Renmin University of China | Neurips /2019                                   | 先用虚拟的数据和标签得到虚拟模型梯度，计算虚拟模型梯度和真正的梯度之间的L2损失，损失求导使得虚拟梯度拟合真实梯度，从而使得虚拟数据和虚拟标签同步变化，最后复现出原始数据。该方案仅适用于单层网络模型。 | 
| Towards Security Threats of Deep Learning Systems: A Survey                             | Yingzhe He                 | IEEE Transactions on Software Engineering /2021 | 深度学习下的攻击综述，包括模型窃取攻击，模型反演攻击，投毒攻击，对抗性样本攻击                                                              |                                                                                               | 
| Local and Central Differential Privacy for Robustness and Privacy in Federated Learning | Openmind                   | NDSS /2021                                      | 联邦场景下LDP和CDP对推理攻击（隐私性）和后门攻击（鲁棒性）的防御研究，内有大量实验设置和对比结果。DP之所以能防御后门攻击可能在于加DP的同时同步给置入后门的攻击方加了噪声，削弱了攻击方能力   |                                                                                               | 
| Comprehensive Privacy Analysis of Deep Learning | University of Massachusetts Amherst                   | NDSS /2018                                     | 在联邦场景下假设联邦中心方或其他参与方是敌手的情况下的成员推理攻击，并分了无监督，半监督，白盒，黑盒，积极攻击，消极攻击等多种场景。主要在于它的成员推理攻击比较普适，攻击的模型思想比较清晰。  |                                                                                               | 
| Evaluating Differentially Private Machine Learning in Practice       | openmind            | CCS /2019                                     | 对不同的DP（高级组合，ZCDP，RDP等）进行评估，用黑盒和白盒的成员推理攻击来评估经过这些DP训练后的模型的防御效果。  |                                                                                              

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
# Code
