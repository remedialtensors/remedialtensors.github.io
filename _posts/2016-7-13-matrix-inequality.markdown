---
layout: post
title: "Using a matrix equality for (small-scale) image classification"
date: 2016-07-20
visible: 1
categories: ml
comments: true
author: "vaishaal"
author_full_name: "Vaishaal Shankar"
---

In this post I will walk through a concrete application of [a matrix equality](http://people.eecs.berkeley.edu/~stephentu/blog/matrix-analysis/2016/06/03/matrix-inverse-equality.html) to speed up the training process of a simple image classification pipeline.


#### Background
I have a relatively small collection of blurry images (32 x 32 rgb pixels) from the cifar data set (50,000 images) from each of 10 classes. The task is to build a
model to classify these images.

For reference, the images look like this:
<p>
<img src="{{site.baseurl}}/assets/images/cifar_frog.png" width="256" id="cifar_frog">
</p>

#### Problem Formulation
We can represent each image as a vector in $\mathrm{R}^{32 \times 32 \times 3}$ (a 3072 dimensional vector).
Then we stack all these images into a $50000 \times 3072$ matrix and call it $X$

We can let Y be a $50000 \times 10$ matrix of corresponding [one hot encoded](http://stackoverflow.com/questions/17469835/one-hot-encoding-for-machine-learning) image labels.

We denote $X_{i}$ to be the $ith$ row of $X$ (or the $ith$ image)

We denote $Y_{i}$ to be the $ith$ row of $Y$ (or the $ith$ image label)

<p>
Now the task is to build a <i>generalizable</i> map from $X_i \to Y_i$
</p>

#### Strawman

What is the simplest map we can think of?

LINEAR!

That is we want to find a matrix $W$ such that $xW$ is close to $y$, the label vector.

Particularly we want the maximal index of $xW$ to be the same as the maximal index
of $y$.

Another way to state this is that we want $\|\|xW - y\|\|_{2}^{2}$ to be small.

We can formulate an optimization problem.

<p>
So lets try to minimize
$\frac{1}{2} \|XW - Y\|_{F}^{2} + \lambda \|W\|^{2}_{F}$
</p>

Note I added a *penalty* term, this is very common.
In the derivation of the solution it will be clear why the penalty
term is necessary. Note the $_F$ simply means I will be using the [Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm),
which means I'll treat the matrices XW and Y as large vectors and use the standard euclidean
norm.

Note:
$\|\|X\|\|_{F}^{2} = \mathrm{Tr}(X^{T}X)$

Where $\mathrm{Tr}$ is the trace.


#### Strawman solution
We can find the optimum solution with some matrix calculus:



First expand

$ \mathrm{Tr}(W^{T}X^{T}XW) - \mathrm{Tr}(X^{T}WY) + \mathrm{Tr}(Y^{T}Y)  + \lambda \mathrm{Tr}(W^{T}W)$

Note I converted the Frobenius norm to a trace

Then take derivative and set to 0.

Note trace and derivative commute

$ \mathrm{Tr}(X^{T}XW) - \mathrm{Tr}(X^{T}Y) + \lambda I_{d} \mathrm{Tr}(W) = 0$

$ \mathrm{Tr}(X^{T}XW) +  \lambda \mathrm{Tr}(W) = \mathrm{Tr}(X^{T}Y)$

Linearity of \mathrm{Tr}ace

$ \mathrm{Tr}((X^{T}X +  I_{d}\lambda) W) = \mathrm{Tr}(X^{T}Y)$

This is satisfied when:

$W =  (X^{T}X +  I_{d}\lambda)^{-1}X^{T}Y$

Which would be our optimum. Since $d$ in this case is only $3072$ this is a quick and easy computation. Note the penalty term makes our matrix invertible when $X^{T}X$
is singular.
that can be done in one line of python.

```
>>> d = X.shape[1]
>>> W = scipy.linalg.solve(X.dot(X) + lambdav*np.eye(d), X.T.dot(Y))
>>> predictedTestLabels = argmax(Xtest.dot(W), axis=1)
```
####How did it do?
```
>>> (predictedTestLabels == labels)/float(len(labels))
0.373
```

...Ouch

####Strawman++

Unfortunately the raw space  these image-vectors live in isn't very good for
linear classification, so our model will perform poorly. So lets "lift" our data
to a "better" space.

Let $\Phi$ be a featurization function, that will "lift" our data. I've
heard neural networks work well for this task, so I'll let $\Phi$ be a convolutional neural net (cnn)

I'm lazy so I don't have time to add a lot of layers, so it'll be a one layer CNN.


Furthermore I'm really lazy so I'm not going to train the network.  So $\Phi$ is a
*random*, *single layer* convolutional neural network.

<p>
Specifically I used a network with $6 x 6$ patches, $1024$ filters, a RELU nonlinearity and a average pooler to $2 x 2$.
This will make the output dimension $4096$
</p>

#### How did it do?

```
>>> A = phi(X, seed=0)
>>> d = A.shape[1]
>>> lambdav = 0.1
>>> W = scipy.linalg.solve(A.T.dot(A) + lambdav*np.eye(d), A.T.dot(Y))
>>> predictedTestLabels = argmax(phi(Xtest, seed=0).dot(W), axis=1)
>>> (predictedTestLabels == labels)/float(len(labels))
0.64
```

Holy smokes batman!

thats a big jump. But we can do better.

####Tinman
Since our $\Phi$ is just a random map, what if we "lift" X
multiple times (independently) and concatenate those representations,
since each one is independent and random.

Let $\Phi_{k}$ be this concatenation map


That is:


```
>>> k = 25
>>> def phik(X):
        return np.hstack(map(lambdav i: phi(X, seed=i), range(k)))
```

We can let $A = \Phi_{k}(X)$, note A is now $50000 \times 4096k$


<p>
Remember we want to minimize $\frac{1}{2}\|AW - Y\|_{F}^{2} + \lambda\|W\|_{F}^{2}$
</p>

Our previous calculus tells us $W^{*} = (A^{T}A + \lambda I_{4096k})^{-1}A^{T}Y$

And our prediction vector would be $\hat{Y} = A(A^{T}A + \lambda I_{d})^{-1}A^{T}Y$

Even for moderate values of k (perhaps over $25$), $(AA^{T} + \lambda I_{d})^{-1}$ becomes very hard to compute (since the inverse scales as $d^{3}$).

We can finally use the [useful matrix equality](https://people.eecs.berkeley.edu/~stephentu/blog/matrix-analysis/2016/06/03/matrix-inverse-equality.html)
to rewrite the prediction vector

$\hat{Y} = AA^{T}(AA^{T} + \lambda I_{50000})^{-1}Y$

Thus our new linear model looks like:

$W = A^{T}(AA^{T} + \lambda I_{50000})^{-1}Y$

Now we never have to invert anything greater than $50000 \times 50000$ !

I'm going to try $k=25$

####How did it do?

```
>>> A = phik(X)
>>> W = A.t.dot(scipy.linalg.solve(A.dot(A.t) + lambdav * np.eye(n), Y, sym_pos=True))
>>> predictedTestLabels= np.argmax(phik(Xtest).dot(C), axis=1)
>>> (predictedTestLabels == labels)/float(len(labels))
0.75
```

yay!

#### You skipped some steps!

There are a couple key details I left out of this post. Both are issues around
making the above method practical (even on small datasets like CIFAR-10).

One is the actual efficient computation of $\Phi$, this step can
be easily parallelized or sped up using vector operations (or both).

The actual observed behavior is that the test accuracy climbs as the number of
random features are accumulated, so we want to push $k$ as large as possible.
But we also want to avoid memory problems when $n \times d$ gets too large.
So we want to avoid materializing X.

