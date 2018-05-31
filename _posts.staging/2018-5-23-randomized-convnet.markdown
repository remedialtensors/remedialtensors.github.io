---
layout: post
title: "Randomizing the layers of a convolutional neural network"
date: 2018-06-04
visible: 0
categories: ml
author: "achalvaishaal"
author_full_name: "Achal Dave and Vaishaal Shankar"
comments: true
---

In this post we will investigate the effect of __randomizing__ layers in a convolutional neural network during training.
More precisely we compare the validation accuracy of image classification models with certain layers __fixed__ to be random
samples from a gaussian distribution.

