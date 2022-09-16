# Answers of [all topics](https://github.com/MsterDC/CVM-DL_Base/blob/main/topic.md) in deep learning

Please submit your answers here

## 🔖 Example

[5] 解释 Focal Loss 与 Center Loss
- [参考论文1](xxx.pdf) | [参考论文2](xxx.pdf) | [参考链接1](xxx.com) | [参考链接2](xxx.com)
```
1. Focal Loss
   one-stage目标检测尽管有着运算速度快的优点，但在性能方面往往低于two-stage的检测器。
   有人发现，导致这种情况的原因是单阶段目标检测器没有对proposal进行筛选，从而导致结果的前后景类别不平衡。
   大量的简单背景proposal尽管单个的loss较低，但累加起来却overwhelm高loss的难样本。
   Focal Loss在交叉熵loss前加上（1-pt)，pt为预测的logit。当样本越简单，pt越大，（1-pt）越小，loss越小，则简单样本的影响越低。
    
2. Center Loss
   以往使用CE进行训练的分类任务，往往只能得到一个分离的特征表示。
   center loss为每一个样本学习一个特征中心，使得每个类别的特征尽可能聚集，产生更加具有判别性的特征。
```

[11] 梯度下降方法找到的一定是目标函数下降最快的方向么？如何理解 GD 与 SGD、minibatchSGD 算法的差异?
```
1. 梯度下降方法找到的一定是目标函数下降最快的方向么？
   不是。
   它只是目标函数在当前的点的切平面上下降最快的方法，只有linear或subliner的速度。
   或者说，局部最优解的方向对于全局最优解来说不一定是最优的。
    
2. 如何理解 GD 与 SGD、minibatchSGD 算法的差异?
   GD是整个training set计算一次梯度，minibatchSGD是每个batch计算一次，SGD是每个样本计算一次。
   随着计算梯度的样本数增加，计算出来的梯度越接近于真实的梯度方向，但速度也越慢，同时没有了噪声的扰动，会难以走出局部最优解。
```
[12]：Batch的大小对于模型收敛的影响？Adam optimizer与SGD的优缺点对比？

[参考链接1](https://www.zhihu.com/question/32673260) | [参考链接2](https://zhuanlan.zhihu.com/p/22252270) | [参考链接3](https://blog.csdn.net/yinyu19950811/article/details/90476956)
```
1. Batch的大小对于模型收敛的影响

首先考虑一种极端情况，每次只训练一个样本，即 Batchsize = 1，训练时每次修正方向以各自样本的梯度方向修正，横冲直撞各自为政，
难以达到收敛;

在合理范围内，增大Batchsize，其确定的下降方向越准，引起训练震荡越小，有助于收敛的稳定性，跑完一次epoch所需的迭代次数减少，
对于相同数据量的处理速度进一步加快;

进一步增大Batchsize，跑完一次epoch所需的迭代次数进一步减少，模型的性能会下降，模型的泛化能力也会下降。
Hoffer等人的研究表明[https://proceedings.neurips.cc/paper/2017/file/a5e0ff62be0b08456fc7f1e88812af3d-Paper.pdf]，
大的Batchsize性能下降是因为训练时间不够长，本质上并不是Batchsize的问题，在同样的epochs下的参数更新变少了，
因此需要更长的迭代次数，要想达到相同的精度，其所花费的时间大大增加了，从而对参数的修正也就显得更加缓慢。
Batchsize增大到一定程度，其确定的下降方向已经基本不再变化（在样本多的情况下，下降方向差异不大）     

2. Adam optimizer与SGD的优缺点对比

对于SGD，选择合适的learning rate比较困难，因为SGD对所有的参数更新使用同样的learning rate。对于稀疏数据或者特征，
有时对于不经常出现的特征我们可能想更新快一些，对于常出现的特征更新慢一些，这时候SGD就不太能满足要求了。
SGD收敛速度慢，容易收敛到局部最优，并且在某些情况下可能被困在鞍点。   

Adam (Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。
Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳，提升训练速度。
```
---

[14] batch normalization 的具体流程? 解决什么问题? 使用时注意事项?训练测试有和差别？
```
1. batch normalization 的具体流程?
   BN是一个先归一化，再逆归一化的过程。假设输入数据的维度是[B, C, W, H]。
   首先计算同一个batch内各个通道（b*w*h个数）的均值和方差，并对每个通道进行归一化。
   同时，每个通道有两个可训练的参数：再缩放参数α和再平移参数β，使用这个参数数据进行逆归一化。
    
2. 解决什么问题?
   从理论上来看，提出BN的文章指出，当模型每一层的数据分布不一致时，会导致训练的难度增加，BN就可以缓解这种分布不一致的现象。
   但后续的文章指出这种不一致是不存在的，真正导致BN生效的原因是L范数的平滑。
   不过BN的理论性还是一个广泛讨论的问题，没有真正的结论；
   从炼丹来看，BN加快网络的训练与收敛速度，控制梯度爆炸防止梯度消失。
   同时BN的计算是在batch层面，所以引入了噪声，缓解了过拟合现象。
   
3. 使用时注意事项?训练测试有和差别？
   需要注意batch的大小对于结果的影响。
   训练时，每个batch的均值和方差是根据输入的batch计算的，比较好获得。
   但是测试时只有一个输入，无法获得batch的数据。
   因此，训练时会每个通道维护一个全局的均值和方差，用于测试时使用。
```

---

**[16]：解释 label smoothing [[参考链接](https://www.cnblogs.com/irvingluo/p/13873699.html)]**

ps:加载不出来公式的同学请参考链接或下载[github公式显示插件](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related)

标签平滑（Label smoothing），像L1、L2和dropout一样，是机器学习领域的一种正则化方法，通常用于分类问题，
目的是防止模型在训练时过于自信地预测标签，改善泛化能力差的问题。

对于分类问题，我们通常认为训练数据中标签向量的目标类别概率应为1，非目标类别概率应为0。传统的one-hot编码的标签向量$y_i$为，

$$
y_i=\begin{cases}  
1, & i=target \\
0, & i\neq target  
\end{cases}
$$

在训练网络时，最小化损失函数$H(y,p)=-\sum_{i}^Ky_ilogp_i$，其中$p_i$由对模型倒数第二层输出的logits向量$z$应用softmax函数计算得到，

$$
p_i=\frac{exp(z_i)}{\sum_{j}^Kexp(z_j)}
$$

传统one-hot编码标签的网络学习过程中，鼓励模型预测为目标类别的概率趋近1，非目标类别的概率趋近0，即最终预测的logits向量（logits向量经过softmax后输出的就是预测的所有类别的概率分布）中目标类别$z_i$的值会趋于无穷大，使得模型向 预测正确与错误标签的logit差值无限增大的方向 学习，而过大的logit差值会使模型缺乏适应性，对它的预测过于自信。在训练数据不足以覆盖所有情况下，这就会导致网络过拟合，泛化能力差，而且实际上有些标注数据不一定准确，这时候使用交叉熵损失函数作为目标函数也不一定是最优的了。

label smoothing的数学定义：label smoothing结合了均匀分布，用更新的标签向量 $\hat{y_i}$ 来替换传统的one-hot编码的标签向量$y_{hot}$:

$$
\hat{y_i}=y_{hot}(1-\alpha)+\alpha/K
$$

其中K为多分类的类别总个数，$\alpha$是一个较小的超参数（一般取0.1），即

$$
\hat{y_i}=\begin{cases}  
1-\alpha, & i=target \\
\alpha/K, & i\neq target  
\end{cases}
$$

这样，标签平滑后的分布就相当于往真实分布中加入了噪声，避免模型对于正确标签过于自信，使得预测正负样本的输出值差别不那么大，从而避免过拟合，提高模型的泛化能力。

---

[18] 池化层有何作用？
```
   特征不变性。由于池化层关注的是某一区域的特征而不是某一特征，所以存在容忍性，输入有微小偏差时不会导致特征的变化。
   特征降维。专人做专事，让卷积操作专心聚合特征，扩大感受野的任务由池化层来完成。
   在一定程度上防止过拟合。
```


[20] 梯度剪裁的含义及目的？
```
   当网络中的每一层的梯度累乘，随着层数的增加，梯度可能越来越到，最后导致梯度爆炸。
   梯度裁剪对于梯度设定一个范围，clip掉过大的梯度，从而防止梯度爆炸。
```


[36] 简述基于通道重要性评估的剪枝方法流程？
```
   基于通道重要性评估的剪枝方法流程由五个部分组成：1.网络初始化 2.加入通道正则的训练 3.通道剪枝 4.微调 5. 获得剪枝模型
   第一步网络初始化是不加入剪枝策略，正常训练模型；
   第二步使用通道的正则约束，强制使不同通道的重要性产生差异；
   第三步将第二步训练好的模型中重要性低的通道直接删去；
   第四步微调剪枝后的模型；
   最后就获得了苗条“的模型。
   其中，2-4步可以多次循环，获得更精简的模型。
```

---

[39]：卷积层参数跟什么有关？影响输出特征图大小的因素包括？
```
卷积层参数和卷积核大小还有个数有关，影响输出特征图大小的因素包括输入特征图大小、卷积核大小、padding的数目、步长
```

---

**[41]：使用 1 次 7x7 卷积的模型和使用 3 次 5x5 卷积的模型在性能、感受野和参数量上是否一样？[[参考链接](https://blog.csdn.net/BigData_Mining/article/details/104418748/)]**


ps:加载不出来公式的同学请参考链接或下载[github公式显示插件](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related)

在卷积神经网络中，感受野（Receptive Field）的定义是卷积神经网络每一层输出的特征图（feature map）上的像素点在输入图片上映射的区域大小。
再通俗点的解释是，特征图上的一个点对应输入图上的区域。

在计算感受野时，最后一层（卷积层或池化层）输出特征图感受野的大小等于卷积核的大小。第i层卷积层的感受野大小和第i层的卷积核大小和步长有关系，
同时也与第（i+1）层感受野大小有关。计算感受野的大小时忽略了图像边缘的影响，即不考虑padding的大小。

关于感受野大小的计算方式是采用从最后一层往下计算的方法，即先计算最深层在前一层上的感受野，然后逐层传递到第一层，使用的公式可以表示如下：
$$
RF_i=(RF_{i+1}-1)\times stride_i+Ksize_i
$$
其中，$RF_i$是第i层卷积层的感受野，$RF_{i+1}$是 $i+1$层上的感受野，$stride$是卷积的步长，$Ksize$是本层卷积核的大小。

一次$7\times7$卷积的感受野为$7\times7$，三次$5\times5$卷积的感受野为$13\times13$（$RF_3=5，RF_2=9，RF_1=13$），所以感受野不同。

参数主要存在于卷积核中，与卷积核尺寸和个数有关，要看具体的网络设计，因此一次$7\times7$卷积和三次$5\times5$卷积的参数量一般情况下是不同的。

关于性能，没有找到具体的衡量指标，但直觉来讲应该是不同的，感受野不同，提取的特征就会有所不同，对性能会有一定的影响

---

[52] mAP 的计算方法？mAP@0.25 与 mAP@0.5 的区别？
```
   mAP是用来衡量目标检测准确性的一种定量指标。
   首先根据预测出的框与GT的IOU进行筛选，可以将结果分为TP，FP，TF，NF四种类型。由TP/(TP+FP)计算出准确率P，由TP/(TP+FN)算出召回率R。
   以P为纵坐标，R为横坐标构建P-R曲线。
   P-R曲线下的面积就是AP，mAP就是各个类别的均值。
   mAP@x，这里的x指的是IoU的阈值设定。
```

---

[62]：什么是知识蒸馏

[参考论文](https://link.springer.com/article/10.1007/s11263-021-01453-z)

```
知识蒸馏属于模型轻量化的一种方法，存在一个大的教师模型，和一个轻量化的学生模型，知识蒸馏的任务就是
迁移教师模型中的知识到学生模型中，指导学生模型的训练提高性能。

基本的两种形式是：
- 基于logits的蒸馏（利用教师的logits指导学生的训练）；
- 基于feature map的蒸馏（迁移教师feature map中的知识）。

随着知识蒸馏的发展，应用范围也被扩展，不只局限于轻量化模型，更多的是看作一种指导知识指导学生模型训练，
学生模型并不一定是一个轻量化的模型。与此同时知识蒸馏也发展出了更多的范式，详细的可以参考链接中的综述。
```

---

**[64]：什么是元学习 [[参考链接](https://zhuanlan.zhihu.com/p/136975128)]**

元学习Meta Learning，含义为学会学习，即learn to learn。Meta Learning希望使得模型获取一种“学会学习”的能力，
使其可以在获取已有“知识”的基础上快速学习新的任务

通常在机器学习里，我们会使用某个场景的大量数据来训练模型；然而当场景发生改变，模型就需要重新训练。
但是对于人类而言，一个小朋友成长过程中会见过许多物体的照片，某一天，当Ta（第一次）仅仅看了几张狗的照片，
就可以很好地对狗和其他物体进行区分。

<font color="#dd0000">元学习中要准备许多任务来进行学习，而每个任务又有各自的训练集和测试集。</font>

以一个具体的任务说明：
有一个图像数据集叫[Omniglot](https://github.com/brendenlake/omniglot)，包含1623个不同的火星文字符，
每个字符包含20个手写的case。这个任务是判断每个手写的case属于哪一个火星文字符。

如果我们要进行N-ways，K-shot（数据中包含N个字符类别，每个字符有K张图像）的一个图像分类任务。比如20-ways，1-shot分类的意思是说，
要做一个20分类，但是每个分类下只有1张图像的任务。我们可以依据Omniglot构建很多N-ways，K-shot任务，这些任务将作为元学习的任务来源。
构建的任务分为训练任务（Train Task），测试任务（Test Task）。特别地，每个任务包含自己的训练数据、测试数据，在元学习里，
分别称为**Support Set**和**Query Set**。

元学习的目的是获取一组更好的模型初始化参数（即让模型自己学会初始化）。我们通过（许多）N-ways，K-shot的任务（训练任务）进行元学习的训练，
使得模型学习到“先验知识”（初始化的参数）。这个“先验知识”在新的N-ways，K-shot任务上可以表现的更好。算法具体流程如下：

<img src="元学习流程.jpg" width = "100" height = "100" div align=center />

Ps：在“预训练”阶段，也可以sample出1个batch的几个任务，那么在更新meta网络时，要使用sample出所有任务的梯度之和。

---

[70] 什么是泛化、鲁棒？
```
   鲁棒性：模型对于输入扰动或对抗样本的能力。
   泛化性：模型对于新数据做出准确预测的能力。
```

---

[81]：CNN 类结构与 Transformer 类结构有什么典型区别

[参考链接](https://blog.csdn.net/qq_39478403/article/details/121099094?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-121099094-blog-118089613.pc_relevant_multi_platform_whitelistv4&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-121099094-blog-118089613.pc_relevant_multi_platform_whitelistv4&utm_relevant_index=2)

```
一方面，CNN 的 Conv 算子存在感受野较局限的问题，为扩大网络的关注区域，需多层堆叠卷积-池化结构，但随之带来的问题是
“有效/真实” 感受野以某个中心为原点向外高斯衰减，因此 CNN 通常的有效Attention就是图中某一两个较重要的Parts。
为解决该问题，可设计用于CNN的Attention Module来得到感受野范围更大而均衡的Attention Map，其有效性得到了很多工作的证明。
另一方面，Transformer天然自带的Long Range特性 (自注意力带来的全局感受野) 使得从浅层到深层特征图，都较能利用全局的有效信息，
并且Multi-Head机制保证了网络可关注到多个Discriminative Parts(每个Head都是一个独立的Attention)，这是Transformer与CNN主要区别之一。
```

---

## 🎃 Answers

[1] xxx
- [xxx](yyy) | [xxx](yyy)
```
1. xxx
    - yyy
2. xxx
    - yyy
```

---

[2] xxx
- [xxx](yyy) | [xxx](yyy)

```
1. xxx
    - yyy
2. xxx
    - yyy
```

---


