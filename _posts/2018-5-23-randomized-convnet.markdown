---
layout: post
title: "Randomizing the layers of a convolutional neural network"
date: 2016-07-20
visible: 0
categories: ml
author: Vaishaal Shankar and Achal Dave
comments: true
---

In this post we will investigate the effect of __randomizing__ layers in a convolutional neural network during training.
More precisely we compare the validation accuracy of image classification models with certain layers __fixed__ to be random
samples from a gaussian distribution.

