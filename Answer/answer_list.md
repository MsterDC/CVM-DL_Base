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


