# 各轮题目分配情况

## 🔰 第一轮题目分配

### 🐶 Common Set

* 吴可鑫 => ['Q12：Batch size 的对于模型收敛的影响？Adam optimizer 与 SGD 的优缺点对比？', 'Q16：解释 label smoothing', 'Q39：卷积层参数跟什么有关？影响输出特征图大小的因素包括？', 'Q41：使用 1 次 7x7 卷积的模型和使用 3 次 5x5 卷积的模型在性能、感受野和参数量上是否一样？', 'Q64：什么是元学习', 'Q81：CNN 类结构与 Transformer 类结构有什么典型区别']

* 王蕊 => ['Q9：逻辑回归用于解决什么问题？以逻辑回归问题为例，训练时不进行输入特征归一化会导致无法 收敛么？进行归一化的目的是什么？', 'Q25：简述 FCN 的核心思想', 'Q32：什么是转置卷积？', 'Q46：有什么解决模型过拟合的途径？', 'Q63：什么是多示例学习', 'Q83：什么是正确的炼丹流程！']

* 魏华鹏 => ['Q23：简述 AlexNet、VGGNet、GoogleNet 以及 ResNet、DenseNet 的结构和特点。', 'Q26：简述 Inception、Xception 及 MobileNet 的核心结构', 'Q31：什么是组卷积？简述 ResNext、ShuffleNet 基本结构', 'Q34：空洞卷积的结构以及存在的问题，什么是膨胀系数？', 'Q65：什么是持 续学习', 'Q78：什么是 Zero/Few shot']

* 董澍威 => ['Q1：激活函数对于神经网络的意义，线性函数是否可以作为激活函数？sigmoid 的基本定义及导数', 'Q8：如何处理非平衡数据集？', 'Q17：BN、LN、GN、IN 有何区别？AdaIN 的使用方法及作用', 'Q22：finetuning 的流程及注意事项（以分类任务为例）', 'Q44：Jaccob 矩阵，Hessian 矩 阵和 Gram 矩阵的公式？', 'Q67：什么是自监督学习']

* 陈东 => ['Q24：简述 NiN 的核心思想，你知道的 1*1 卷积一般都有何作用？', 'Q37：训练和测试原始 VGG19 网络时，为何要固定网络输入大小？如果使用了不同大小的输入图 像，会出现什么问题？', 'Q38：ResNet18 有多少卷积层？', 'Q42：cross entropy loss, hinge loss, MSE loss 的公式？鲁 棒性分别如何？', 'Q43：分类任务为什么不使用 MSEloss？使用了会有什么问题？', 'Q66：什么是半监督学习、什么是弱监督学习，典型应用']

* 陈永杰 => ['Q7：梯度消失和梯度爆炸产生的原因是什么的？如何确定模型发生了梯度爆炸？有什么缓解办法？', 'Q13：学习率过大对训练过程有何影响？什么是学习率调度策略？如何选择？', 'Q29：简述 FPN 的基本结构', 'Q33：简述 Attention 及 Channel attention 的计算过程', 'Q49：图像相关问题训练时预处理(增强)手段有哪些? 应用场景各是怎么样', 'Q80：什么是基于优化的方法，与 feedforward 方法有何区别，有什么典型应用']

* 任海瑞 => ['Q3：简述 Softmax 的定义及意义。', 'Q4：简述 L/A/Am Softmax、GroupSoftMax 的定义及意义。', 'Q10：模型的 bias 与 variance 的区别？如何降低？', 'Q40：对于 VGG16 来说，当图像尺寸变为 2 倍，卷积层的参数数量变为几倍？', 'Q45：怎么对模型效果进行评估的？当机器学习性 能遭遇瓶颈时，你会如何优化？', 'Q56：什么是长尾问题']

* 孔笑宇 => ['Q5：解释 Focal Loss 与 Center Loss。', 'Q11：梯度下降方法找到的一定是目标函数下降最快的方向么？如何理解 GD 与 SGD、minibatchSGD 算法的差异?', 'Q14：batch normalization 的具体流程? 解决什么问题? 使用时注意事项?训练测试有和差别？', 'Q18：池化层有何作用？', 'Q20：梯度剪裁的含义及目的？', 'Q70：什么是泛化、鲁棒']

* 贾云兵 => ['Q2：ReLU 的全程是什么？优缺点？写出几种常见 ReLU 的形式， PReLU 在反向传播中如何处理？', 'Q19：Dropout 的含义、目的，测试时和训练时有何区别？同一种模型结构训练时使用不同的 dropout 比率，测试时速度有何对应变化？', 'Q28：简述 UNet 的基本结构', 'Q30：简述 SENet 的核心结构', 'Q35：什么是可变形卷积，如何实现，有何作用', 'Q82：什么是 SOTA、vanilla、oracle、benchmark、head、neck、bottleneck、backbone、 embedding、logits、pretext/downstream task']

* 周梓俊 => ['Q6：模型过拟合问题该如何解决？', 'Q15：解释 warmup', 'Q21：模型参数初始化对训练结果是否有影响？常用的初始化策略是什么？', 'Q27：简述 EfficientNet 的核心思想', 'Q79：什么是 domain adapatation']

### 🐱Specific Set

* 吴可鑫 => [Q47：TP, FP, TN, FN 的定义？准确率 (accuracy), 精准率 (precision), 召回率 (recall) 定义？PR 曲线和 ROC 曲线的坐标？如果 A 模型的 PR 曲线被 B 模型的 PR 曲线完全包住，哪个模型性能好？，Q61：什么是 NeRF]

* 王蕊 =>[Q51：IOU 如何计算？，Q73：什么是流模型，有什么局限]

* 魏华鹏 =>[Q50：RCNN、Fast RCNN 和 Faster RCNN 的区别与联系？YOLO 与 Faster RCNN 的区别，Q77：什么是类激活图]

* 董澍威 =>[Q60：什么是图像重定向，Q75：什么是 GAN 反演，有几种方法]

* 陈东 =>[Q54：什么是 LPIPS，Q68：什么是场景图，Q74：什么是变分自编码器]

* 陈永杰 =>[Q62：什么是知识蒸馏，Q72：什么是扩散模型]

* 任海瑞 =>[Q48：SSIM、PSNR 的含义及具体定义，Q53：什么是 TV loss，Q76：什么是伪造检测]

* 孔笑宇 =>[Q36：简述基于通道重要性评估的剪枝方法流程，Q52：mAP 的计算方法？mAP@0.25 与 mAP@0.5 的区别？]

* 贾云兵 =>[Q57：什么是开集识别，Q59：什么是超分，什么是盲超分，Q71：什么是 WGAN、CycleGAN、StyleGAN、StarGAN]

* 周梓俊 =>[Q55：什么是 FID，Q58：什么是风格化，Q69：什么是 Prompt learning]


## 🔰 第二轮题目分配
xxx
### 🐶 Common Set
xxx
### 🐱Specific Set
xxx
