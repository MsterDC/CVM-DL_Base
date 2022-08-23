## 深度学习基础自查
1. 激活函数对于神经网络的意义，线性函数是否可以作为激活函数？ sigmoid 的基本定义及导数
2. ReLU 的全程是什么？优缺点？写出几种常见 ReLU 的形式， PReLU 在反向传播中如何处理？
3. 简述 Softmax 的定义及意义。
4. 简述 L/A/Am Softmax 、 GroupSoftMax 的定义及意义。
5. 解释 Focal Loss 与 Center Loss 。
6. 模型过拟合问题该如何解决？
7. 梯度消失和梯度爆炸产生的原因是什么的？如何确定模型发生了梯度爆炸？有什么缓解办法？
8. 如何处理非平衡数据集？
9. 逻辑回归用于解决什么问题？以逻辑回归问题为例，训练时不进行输入特征归一化会导致无法
收敛么？进行归一化的目的是什么？
10. 模型的 bias 与 variance 的区别？如何降低？
11. 梯度下降方法找到的一定是目标函数下降最快的方向么？如何理解 GD 与 SGD 、 minibatchSGD
算法的差异
12. Batch size 的对于模型收敛的影响？ Adam optimizer 与 SGD 的优缺点对比？
13. 学习率过大对训练过程有何影响？什么是学习率调度策略？如何选择？
14. batch normalization 的具体流程 ? 解决什么问题 ? 使用时 注意事项 训练测试有和差别？
15. 解释 warmup
16. 解释 label smoothing
17. BN 、 LN 、 GN 、 IN 有何区别？ AdaIN 的使用方法及作用
18. 池化层有何作用？
19. Dropout 的含义、目的，测试时和训练时有何区别？同一种模型结构训练时使用不同的
dropout 比率，测试时速度有何对应变化？
20. 梯度剪裁的含义及目的？
21. 模型参数初始化对训练结果是否有影响？常用的初始化策略是什么？
22. finetuning 的流程及注意事项（以分类任务为例）
23. 简述 AlexNet 、 VGGNet 、 GoogleNet 以及 ResNet 、 DenseNet 的结构和特点。
24. 简述 NiN 的核心思想，你知道的 1*1 卷积一般都有何作用？
25. 简述 FCN 的核心思想
26. 简述 Inception 、 Xception 及 MobileNet 的核心结构
27. 简述简述EfficientNetEfficientNet的核心思想的核心思想
28. 简述简述UNetUNet的基本结构的基本结构
29. 简述简述FPNFPN的基本结构的基本结构
30. 简述简述SENetSENet的核心结构的核心结构
31. 什么是组卷积？简述什么是组卷积？简述ResNextResNext、、ShuffleNetShuffleNet基本结构基本结构
32. 什么是转置卷积？什么是转置卷积？
33. 简述简述AttentionAttention及及Channel attentionChannel attention的计算过程的计算过程
34. 空洞卷积的结构以及存在的问题，什么是膨胀系数？空洞卷积的结构以及存在的问题，什么是膨胀系数？
35. 什么是可变形卷积，如何实现，有何作用什么是可变形卷积，如何实现，有何作用
36. 简述基于通道重要性评估的剪枝方法流程简述基于通道重要性评估的剪枝方法流程
37. 训练和测试原始训练和测试原始VGG19VGG19网络时，为何要固定网络输入大小？如果使用了不同大小的输入图网络时，为何要固定网络输入大小？如果使用了不同大小的输入图像，会出现什么问题？像，会出现什么问题？
38. ResNet18ResNet18有多少卷积层？有多少卷积层？
39. 卷积层参数跟什么有关？影响输出特征图大小的因素包括？卷积层参数跟什么有关？影响输出特征图大小的因素包括？
40. 对于对于VGG16VGG16来说，来说，当图像尺寸变为当图像尺寸变为 22 倍，卷积层的参数数量变为几倍？倍，卷积层的参数数量变为几倍？
41. 使用使用11次次7x77x7卷积的模型和使用卷积的模型和使用33次次5x55x5卷积的模型在性能、感受野和参数量上是否一样？卷积的模型在性能、感受野和参数量上是否一样？
42. crosscross--entropy loss, hinge loss, MSE lossentropy loss, hinge loss, MSE loss的公式？鲁棒性分别如何？的公式？鲁棒性分别如何？
43. 分类任务为什么不使用分类任务为什么不使用MSElossMSEloss？使用了会有什么问题？？使用了会有什么问题？
44. JaccobJaccob矩阵，矩阵，HessianHessian矩阵和矩阵和GramGram矩阵的公式？矩阵的公式？
45. 怎么对模型效果进行评估的？当机器学习性能遭遇瓶颈时，你会如何优化？怎么对模型效果进行评估的？当机器学习性能遭遇瓶颈时，你会如何优化？
46. 有什么解决模型过拟合的途径？有什么解决模型过拟合的途径？
47. TP, FP, TN, FNTP, FP, TN, FN的定义？准确率的定义？准确率 (acc(accuracy),uracy), 精准率精准率 (precision),(precision), 召回率召回率 (recall)(recall) 定义？定义？PP--RR曲线和曲线和ROCROC曲线的坐标？如果曲线的坐标？如果AA模型的模型的PP--RR曲线被曲线被BB模型的模型的PP--RR曲线完全包住，哪个模型性能好？曲线完全包住，哪个模型性能好？
48. SSIMSSIM、、PSNRPSNR的含义及具体定义的含义及具体定义
49. 图像相关问题训练时预处理图像相关问题训练时预处理((增强增强))手段有哪些手段有哪些? ? 应用场景各是怎么样应用场景各是怎么样
50. RCNNRCNN、、FastFast--RCNNRCNN和和FasterFaster--RCNNRCNN的区别与联系？的区别与联系？YOLOYOLO与与FasterFaster--RCNNRCNN的区别的区别
51. IOUIOU如何计算？如何计算？
52. mAPmAP的计算方法？的计算方法？mAP@0.25mAP@0.25与与mAP@0.5mAP@0.5的区别？的区别？
53. 什么是什么是TV lossTV loss
54. 什么是什么是LPIPSLPIPS
55. 什么什么是是FIDFID
56. 什么是长尾问题什么是长尾问题
57. 什么是开集识别什么是开集识别
58. 什么是风格化什么是风格化
59. 什么是超分，什么是盲超分什么是超分，什么是盲超分
60. 什么是图像重定向什么是图像重定向
61. 什么是什么是NeRFNeRF
62. 什么是知识蒸馏什么是知识蒸馏
63. 什么是多示例学习什么是多示例学习
64. 什么是元学习什么是元学习
65. 什么是持续学习什么是持续学习
66. 什么是半监督学习、什么是弱监督学习，典型应用什么是半监督学习、什么是弱监督学习，典型应用
67. 什么是自监督学习什么是自监督学习
68. 什么是场景图什么是场景图
69. 什么什么Prompt learningPrompt learning
70. 什么是泛化、鲁棒什么是泛化、鲁棒
71. 什么是什么是WGANWGAN、、CycleGANCycleGAN、、StyleGANStyleGAN、、StarGANStarGAN
72. 什么是扩散模型什么是扩散模型
73. 什么是流模型，有什么局限什么是流模型，有什么局限
74. 什么是变分自编码器什么是变分自编码器
75. 什么是什么是GANGAN反演，有几种方法反演，有几种方法
76. 什么是伪造检测什么是伪造检测
77. 什么是什么是类激活图类激活图
78. 什么是什么是Zero/Few shotZero/Few shot
79. 什么是什么是domain adapatationdomain adapatation
80. 什么是基于优化的方法，与什么是基于优化的方法，与feedforwardfeedforward方法有何区别，有什么典型应用方法有何区别，有什么典型应用
81. CNNCNN类结构与类结构与TransformerTransformer类结构有什么典型区别类结构有什么典型区别
82. 什么是什么是SOTASOTA、、vanillavanilla、、oracleoracle、、benchmarkbenchmark、、headhead、、neckneck、、bottleneckbottleneck、、backbonebackbone、、embeddingembedding、、logitslogits、、pretext/downstream taskpretext/downstream task
83. 什么是正确的炼丹流程！什么是正确的炼丹流程！
