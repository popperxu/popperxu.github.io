---
layout: post
title: 简单分类器的python代码实现
categories: machine_learning
tags: ml
---

本文是stanford大学课程：Convolutional Neural Networks for Visual Recognition 的一些笔记与第一次作业。主要内容为简单（多类）分类器的实现:KNN, SVM, softmax。

softmax与SVM的一点区别，其中一张PPT说明：
![](softmax.jpg)

linear classifier的实现，需要补全 train 与 predict 两部分：

$$ \sum_{i} a_i$$




然后就是svm与softmax分类器的loss与梯度的实现。

SVM 的loss function 与 gradient ：

loss function: 
$$ L = \frac{1}{N} \sum_i \sum_{y_i \ne j}  \max( 0, \mathrm{f}(\mathrm{x}_i, W)_j - \mathrm{f}(\mathrm{x}_i, W)_{y_i} + 1  ) + \frac{\lambda}{2} \sum_k\sum_l W_{k,l}^2 $$


然后测试代码在IPython Notebook上完成，可以进行单元测试，一点一点运行。

比如，基于给定数据集，KNN的测试结果（K=10）：
