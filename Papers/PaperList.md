# References for Corresponding Topics

## 🌻 Registration Example <br> 

[Q5] 解释 warmup
* 2017ICCV - Focal loss for dense object detection, Lin T Y, Goyal P, Girshick R, et al. [Focal Loss](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)

## 🔥 References

[Q2] ReLU 的全程是什么？优缺点？写出几种常见 ReLU 的形式， PReLU 在反向传播中如何处理？

* 2011JMLR - Deep sparse rectifier neural networks, Glorot, Xavier and Bordes, Antoine and Bengio, Yoshua. [ReLU]( http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf)
* 2013 - Rectifier nonlinearities improve neural network acoustic models, Andrew L. Maas, Awni Y. Hannun and Andrew Y. Ng.[Leaky ReLU](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.693.1422&rep=rep1&type=pdf)
* 2015 - Fast and accurate deep network learning by exponential linear units (elus), Djork-Arne Clevert, Thomas Unterthiner and Sepp Hochreiter.[ELU](https://arxiv.org/pdf/1511.07289.pdf%5cnhttp://arxiv.org/abs/1511.07289%5cnhttp://arxiv.org/abs/1511.07289.pdf)
* 2015ICCV - Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun.[PReLU](https://openaccess.thecvf.com/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)
* 2020SMA - DPReLU: Dynamic Parametric Rectified Linear Unit, Kien Mai Ngoc∗, Donghun Yang,  Iksoo Shin, Hoyong Kim, Myunggwon Hwang†. [DPRelu](https://dl.acm.org/doi/abs/10.1145/3426020.3426049?casa_token=stfS9VciIHsAAAAA:EG5QNTijIOMXeG9XqYjGsN98N7NgVvR6XR88S2KPA-WNBfXMKV5ThUigkL3Bq-dO2tlmN6_jER3V8w)

[Q4] 简述 L/A/Am-Softmax、GroupSoftMax 的定义及意义
* 2016ICML - Large-Margin Softmax Loss for Convolutional Neural Networks, Weiyang Liu, Yandong Wen, Zhiding Yu, Meng Yang. [L-Softmax](https://arxiv.org/pdf/1612.02295.pdf)
* 2017CVPR - SphereFace: Deep Hypersphere Embedding for Face Recognition, Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, Le Song. [A-Softmax](https://arxiv.org/pdf/1704.08063.pdf)
* 2018ICLRW - Additive Margin Softmax for Face Verification, Feng Wang, Weiyang Liu, Haijun Liu, Jian Cheng. [Am-Softmax](https://arxiv.org/pdf/1801.05599.pdf)
* TechReport - GroupSoftmax：利用COCO和CCTSDB训练83类检测器, Chen, Yuntao and and Han, et al. [Group-Softmax](https://zhuanlan.zhihu.com/p/73162940)
* 2020CVPR - Overcoming Classifier Imbalance for Long-tail Object Detection with Balanced Group Softmax, Yu Li, Tao Wang, Bingyi Kang, Sheng Tang, Chunfeng Wang, Jintao Li, Jiashi Feng. [GroupSoftmax](https://arxiv.org/abs/2006.10408)
* 2017PMLR - Efficient softmax approximation for GPUs, Grave, Armand Joulin, Moustapha Cissé, David Grangier, Hervé Jégou. [softmax](http://proceedings.mlr.press/v70/grave17a.html?ref=https://githubhelp.com)

[Q5] 解释 Focal Loss 与 Center Loss
* 2016ECCV - A Discriminative Feature Learning Approach for Deep Face Recognition, Ronneberger, Yandong Wen, Kaipeng Zhang, Zhifeng Li, and Yu Qiao. [Center Loss](https://link.springer.com/content/pdf/10.1007%2F978-3-319-46478-7_31.pdf)
* 2017ICCV - Focal loss for dense object detection, Lin T Y, Goyal P, Girshick R, et al. [Focal Loss](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)

[Q15] 解释 warmup
* 2017CVPR - Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He. [Warmup](https://arxiv.org/pdf/1706.02677.pdf)

[Q17] BN、LN、GN、IN 有何区别？AdaIN 的使用方法及作用
* 2015ICML - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, Sergey Ioffe, Christian Szegedy. [BN](https://arxiv.org/pdf/1502.03167v2.pdf)  <br>
* 2016 - Layer Normalization, Ba, Jimmy Lei and Kiros, Jamie Ryan and Hinton, Geoffrey E. [LN](https://arxiv.org/pdf/1607.06450.pdf)  <br>
* 2018ECCV - Group Normalization, Yuxin Wu, Kaiming He. [GN](https://arxiv.org/pdf/1803.08494.pdf)  <br>
* 2016 - Instance Normalization: The Missing Ingredient for Fast Stylization, Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky. [IN](https://arxiv.org/pdf/1607.08022.pdf)
* 2017ICCV - Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization, Xun Huang and Serge Belongie.[AdaIN](https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf)

[Q19] Dropout 的含义、目的，测试时和训练时有何区别？同一种模型结构训练时使用不同的 dropout 比率，测试时速度有何对应变化？
* 2014JMLR - Dropout: A Simple Way to Prevent Neural Networks from
Overfitting, Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov. [Dropout](https://cs.nju.edu.cn/zhangl/dropout.pdf)
* 2012NEC - Improving neural networks by preventing co-adaptation of feature detectors, Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, Ruslan R. Salakhutdinov. [Dropout](https://arxiv.org/abs/1207.0580)
* 2013 - Improving Neural Networks with Dropout, Nitish Srivastava , Nitish Srivastava. [Dropout](http://www.cs.toronto.edu/~nitish/msc_thesis.pdf)

[Q23] 简述 AlexNet、VGGNet、GoogleNet 以及 ResNet、DenseNet 的结构和特点。
* 2012NIPS - ImageNet Classification with Deep Convolutional Neural Networks, Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E. [AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
* 2015ICLR - Very deep convolutional networks for large-scale image recognition, Simonyan, Karen and Zisserman, Andrew. [VGGNet](https://arxiv.org/pdf/1409.1556.pdf)
* 2015CVPR - Going deeper with convolutions, Szegedy, Christian and Liu, Wei and Jia, Yangqing and Sermanet, Pierre and Reed, Scott and Anguelov, Dragomir and Erhan, Dumitru and Vanhoucke, Vincent and Rabinovich, Andrew. [GoogleNet](https://arxiv.org/pdf/1409.4842.pdf)
* 2016CVPR - Deep residual learning for image recognition, He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian. [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
* 2017CVPR - Densely connected convolutional networks, Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q. [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)

[Q24] 简述 NiN 的核心思想，你知道的 1*1 卷积一般都有何作用？
* 2014ICLR - Network In Network, Lin, Min and Chen, Qiang and Yan, Shuicheng. [NiN](https://arxiv.org/pdf/1312.4400.pdf)

[Q25] 简述 FCN 的核心思想
* 2015CVPR - Fully Convolutional Networks for Semantic Segmentation, Jonathan Long, Evan Shelhamer, Trevor Darrell. [FCN](https://arxiv.org/pdf/1411.4038.pdf)

[Q26] 简述 Inception、Xception 及 MobileNet 的核心结构
* 2015CVPR - Going Deeper with Convolutions, Szegedy C, Liu W, Jia Y, et al. [Inception(v1)](https://arxiv.org/pdf/1409.4842.pdf)
* 2015ICML - Batch normalization: Accelerating deep network training by reducing internal covariate shift, Ioffe S, Szegedy C. [Inception(v2)](http://proceedings.mlr.press/v37/ioffe15.pdf)
* 2016CVPR - Rethinking the inception architecture for computer vision, Szegedy C, Vanhoucke V, Ioffe S, et al. [Inception(v3)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)
* 2017CVPR - Xception: Deep learning with depthwise separable convolutions, François Chollet. [Xception](https://arxiv.org/pdf/1610.02357.pdf)
* 2017CVPR - Mobilenets: Efficient convolutional neural networks for mobile vision applications, Howard, Andrew G and Zhu, Menglong and Chen, Bo and Kalenichenko, Dmitry and Wang, Weijun and Weyand, Tobias and Andreetto, Marco and Adam, Hartwig. [MobileNet](https://arxiv.org/pdf/1704.04861.pdf)
* 2016CVPR Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning, Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke. [Inception-v4](https://arxiv.org/pdf/1602.07261.pdf)

[Q27] 简述 EfficientNet 的核心思想
* 2019ICML - Efficientnet: Rethinking model scaling for convolutional neural networks, Tan, Mingxing and Le, Quoc. [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf)

[Q28] 简述 UNet 的基本结构
* 2015MICCAI - U-Net: Convolutional Networks for Biomedical Image Segmentation, Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas. [U-Net](https://arxiv.org/pdf/1505.04597.pdf)

[Q29] 简述 FPN 的基本结构
* 2017CVPR - Feature pyramid networks for object detection, Lin, Tsung-Yi and Doll{\'a}r, Piotr and Girshick, Ross and He, Kaiming and Hariharan, Bharath and Belongie, Serge. [FPN](https://arxiv.org/pdf/1612.03144.pdf)

[Q30] 简述 SENet 的核心结构
* 2018CVPR - Squeeze-and-Excitation Networks, Jie Hu, Li Shen, Gang Sun. [SENet](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)

[Q31] 什么是组卷积？简述 ResNext、ShuffleNet 基本结构
* 2016 - Deep Roots: Improving CNN Efficiency with Hierarchical Filter Groups, Yani Ioannou, Duncan Robertson, Roberto Cipolla, Antonio Criminisi. [Group Convolution](https://arxiv.org/pdf/1605.06489.pdf)
* 2017CVPR - Aggregated Residual Transformations for Deep Neural Networks, Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He. [ResNext](https://arxiv.org/pdf/1611.05431.pdf)
* 2018CVPR - Shufflenet: An extremely efficient convolutional neural network for mobile devices, Zhang, Xiangyu and Zhou, Xinyu and Lin, Mengxiao and Sun, Jian. [ShuffleNet](https://arxiv.org/pdf/1707.01083.pdf)
* 2018ECCV ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design, Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun. [ShuffleNet v2](https://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html)

[Q32] 什么是转置卷积？
* 2016 - A guide to convolution arithmetic for deep learning, Dumoulin, Vincent and Visin, Francesco. [Transposed Convolution](https://arxiv.org/pdf/1603.07285.pdf)
* 2010CVPR - Deconvolutional networks, Matthew D. Zeiler; Dilip Krishnan; Graham W. Taylor; Rob Fergus. [Deconvolutional networks](https://ieeexplore.ieee.org/abstract/document/5539957)
* 2011CVPR - Adaptive deconvolutional networks for mid and high level feature learning, Matthew D. Zeiler; Graham W. Taylor; Rob Fergus. [Adaptive deconvolutional networks](https://ieeexplore.ieee.org/abstract/document/6126474)

[Q33] 简述 Attention 及 Channel attention 的计算过程
* 2017NIPS - Attention is all you need, Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. [Attention](https://arxiv.org/pdf/1706.03762.pdf)
* 2018ECCV - CBAM: Convolutional Block Attention Module, Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon. [Channel attention](https://arxiv.org/pdf/1807.06521v2.pdf)

[Q34] 空洞卷积的结构以及存在的问题，什么是膨胀系数？
* 2016ICLR - Multi-scale context aggregation by dilated convolutions, Yu, Fisher and Koltun, Vladlen. [Dilated Convolution](https://arxiv.org/pdf/1511.07122v2.pdf)

[Q35] 什么是可变形卷积，如何实现，有何作用
* 2017ICCV - Deformable Convolutional Networks, Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, Yichen Wei. [DCN](https://openaccess.thecvf.com/content_ICCV_2017/papers/Dai_Deformable_Convolutional_Networks_ICCV_2017_paper.pdf)

[Q36] 简述基于通道重要性评估的剪枝方法流程
* 2017ICCV - Learning efficient convolutional networks through network slimming, Liu Z, Li J, Shen Z, et al. [Ref-1](https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)
* 2018Arxiv - Rethinking the smaller-norm-less-informative assumption in channel pruning of convolution layers, Ye J, Lu X, Lin Z, et al. [Ref-2](https://arxiv.org/pdf/1802.00124.pdf)
* 2018ECCV - Data-driven sparse structure selection for deep neural networks, Huang Z, Wang N. [Ref-3](https://openaccess.thecvf.com/content_ECCV_2018/papers/Zehao_Huang_Data-Driven_Sparse_Structure_ECCV_2018_paper.pdf)
* 2017ICCV - Channel pruning for accelerating very deep neural networks, He Y, Zhang X, Sun J. [Ref-4](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Channel_Pruning_for_ICCV_2017_paper.pdf)
* 2018WACV - Learning to Prune Filters in Convolutional Neural Networks, Qiangui Huang; Kevin Zhou; Suya You; Ulrich Neumann. [Ref-5](https://ieeexplore.ieee.org/abstract/document/8354187)
* 2018CVPR - Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks, Yang He, Guoliang Kang, Xuanyi Dong, Yanwei Fu, Yi Yang. [Ref-6](https://arxiv.org/abs/1808.06866)

[Q37] 训练和测试原始 VGG19 网络时，为何要固定网络输入大小？如果使用了不同大小的输入图像，会出现什么问题？
* 2015ICLR - Very deep convolutional networks for large-scale image recognition, Simonyan, Karen and Zisserman, Andrew. [VGGNet]

[Q38] ResNet18 有多少卷积层？
* 2016CVPR - Deep Residual Learning for Image Recognition, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

[Q50] RCNN、Fast-RCNN 和 Faster-RCNN 的区别与联系？YOLO 与 Faster-RCNN 的区别
* 2014CVPR - Rich feature hierarchies for accurate object detection and semantic segmentation, Girshick, Ross and Donahue, Jeff and Darrell, Trevor and Malik, Jitendra. [RCNN](https://arxiv.org/pdf/1311.2524.pdf)
* 2015ICCV - Fast R-CNN, Ross Girshick. [Fast-RCNN](https://arxiv.org/pdf/1504.08083.pdf)
* 2015NIPS - Faster r-cnn: Towards real-time object detection with region proposal networks, Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian. [Faster-RCNN](https://arxiv.org/pdf/1506.01497.pd)
* 2016CVPR - You only look once: Unified, real-time object detection, Redmon, Joseph and Divvala, Santosh and Girshick, Ross and Farhadi, Ali. [YOLO](https://arxiv.org/pdf/1506.02640.pdf)
* 2019IEEE Access - A Survey of Deep Learning-based Object Detection, Licheng Jiao, Fan Zhang, Fang Liu, Shuyuan Yang, Lingling Li, Zhixi Feng, Rong Qu. [Survey of OD](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8825470)

[Q54] 什么是 LPIPS
* 2018CVPR - The unreasonable effectiveness of deep features as a perceptual metric, Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver. [LPIPS](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.pdf)

[Q55] 什么是 FID
* NIPS2017 - GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium, Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter. [FID](https://arxiv.org/pdf/1706.08500.pdf)

[Q58] 什么是风格化
* 2016CVPR - Image Style Transfer Using Convolutional Neural Networks, Leon A. Gatys, Alexander S. Ecker, Matthias Bethge. [NST](https://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

[Q61] 什么是 NeRF
* 2020ECCV - NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, By Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng. (NeRF)[https://dl.acm.org/doi/pdf/10.1145/3503250]

[Q62] 什么是知识蒸馏
* 2015 - Distilling the Knowledge in a Neural Network, Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff [知识蒸馏](https://arxiv.org/pdf/1503.02531.pdf)

[Q71] 什么是 WGAN、CycleGAN、StyleGAN、StarGAN
* 2017ICML - Wasserstein generative adversarial networks, Martin Arjovsky, Soumith Chintala, Léon Bottou. [WGAN](https://arxiv.org/pdf/1701.07875.pdf)
* 2017ICCV - Unpaired image-to-image translation using cycle-consistent adversarial networks, Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A. [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf)
* 2019CVPR - A Style-Based Generator Architecture for Generative Adversarial Networks, Tero Karras, Samuli Laine, Timo Aila. [StyleGAN](https://arxiv.org/pdf/1812.04948.pdf)
* 2018CVPR - Stargan: Unified generative adversarial networks for multi-domain image-to-image translation, Choi, Yunjey and Choi, Minje and Kim, Munyoung and Ha, Jung-Woo and Kim, Sunghun and Choo, Jaegul. [StarGAN](https://arxiv.org/pdf/1711.09020.pdf)
* 2017CVPR - CycleGAN, a Master of Steganography, Casey Chu, Andrey Zhmoginov, Mark Sandler.[CycleGAN](https://arxiv.org/abs/1712.02950)
* 2020CVPR - Analyzing and Improving the Image Quality of StyleGAN, Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila. [StyleGAN](https://openaccess.thecvf.com/content_CVPR_2020/html/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.html)

[Q72] 什么是扩散模型
* 2022NIPS - Denoising diffusion probabilistic models, Ho J, Jain A, Abbeel P. [DiffusionModel](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)

[Q73] 什么是流模型，有什么局限
* 2015ICLR - NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION, Laurent Dinh David Krueger Yoshua Bengio. [FLOW](https://arxiv.org/pdf/1410.8516.pdf)

[Q75] 什么是 GAN 反演，有几种方法
* 2021CVPR - GAN Inversion: A Survey, Weihao Xia, Yulun Zhang, Yujiu Yang, Jing-Hao Xue, Bolei Zhou, Ming-Hsuan Yang. [GAN Inversion](https://arxiv.org/pdf/2101.05278.pdf)

[Q77] 什么是类激活图
* 2016CVPR - Learning deep features for discriminative localization, Zhou, Bolei and Khosla, Aditya and Lapedriza, Agata and Oliva, Aude and Torralba, Antonio. [CAM](https://arxiv.org/pdf/1512.04150.pdf)
* 2020ECCV - Rethinking Class Activation Mapping for Weakly Supervised Object Localization, Wonho Bae, Junhyug Noh, Gunhee Kim. [RCAM](https://link.springer.com/chapter/10.1007/978-3-030-58555-6_37)

[Q83] 什么是正确的炼丹流程
* 2015DPCC - MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems, Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao, Bing Xu, Chiyuan Zhang, Zheng Zhang. [MxNet](https://arxiv.org/abs/1512.01274)
* Zhihu-URL - [炼丹介绍](https://zhuanlan.zhihu.com/p/23781756)
