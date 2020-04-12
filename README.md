# Multi-Task-and-Meta-Learning (Keep Updating)
This repo is for recording interesting papers in Multi-Task and Meta Learning area. This is partly inspired from [CS330: Deep Multi-Task and Meta Learning](https://cs330.stanford.edu/) course in Stanford by Chelsea Finn. 

Welcome contributing to this repo (EXCITED) :smiley:



[toc]

## Multi-Task Learning

End-to-End Multi-Task Learning with Attention. CVPR 2019. [paper](https://arxiv.org/abs/1803.10704)

Latent Multi-task Architecture Learning. AAAI 2019. [paper](https://arxiv.org/abs/1705.08142)

Cross-stitch Networks for Multi-task Learning. CVPR 2016. [paper](https://arxiv.org/abs/1604.03539)

Learning Multiple Tasks with Multilinear Relationship Networks. NIPS 2017. [paper](https://arxiv.org/abs/1506.02117)

## Meta Learning

### Black-Box Approaches

#### Recurrent Neural Network

(MANN) Meta-learning with memory-augmented neural networks. ICML 2016. [paper](http://proceedings.mlr.press/v48/santoro16.pdf)

#### Attention-Based Network

Matching Networks for One-Shot Learning. NIPS 2016. [paper](https://arxiv.org/abs/1606.04080)

(SNAIL)  A Simple Neural Attentive Meta-Learner. ICLR 2018. [paper](https://arxiv.org/pdf/1707.03141.pdf)

### Optimization-Based Methods

(MAML) Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML 2017. [paper](https://arxiv.org/pdf/1703.03400.pdf) :star:

(Reptile; First-order method) On First-Order Meta-Learning Algorithms. arXiv 2018. [paper](https://arxiv.org/pdf/1803.02999.pdf)

#### Other Forms of Prior on MAML

(Implicit MAML) Meta-Learning with Implicit Gradients. NIPS 2019. [paper](https://arxiv.org/abs/1909.04630)

(Implicit Differentiation; SVM) Meta-Learning with Differentiable Convex Optimization. CVPR 2019. [paper](https://arxiv.org/abs/1904.03758)

(Bayesian linear regression) Meta-Learning Priors for Efficient Online Bayesian Regression. Workshop on the Algorithmic Foundations of Robotics 2018. [paper](https://arxiv.org/pdf/1807.08912)

(Ridge regression; Logistic regression) Meta-learning with Differentiable Closed-Form Solvers. ICLR 2019. [paper](https://openreview.net/pdf?id=HyxnZh0ct7) 

#### Understanding MAML

(MAML expressive power and university) Meta-Learning and Universality: Deep Representations and Gradient Descent can Approximate any Learning Algorithm. ICLR 2018. [paper](https://arxiv.org/abs/1710.11622)

(Map MAML to Bayes Framework) Recasting Gradient-Based Meta-Learning as Hierarchical Bayes. ICLR 2018. [paper](https://openreview.net/pdf?id=BJ_UL-k0b) 

#### Tricks to Optimize MAML

**Choose architecture that is effective for inner gradient-step**

Auto-Meta: Automated Gradient Based Meta Learner Search. NIPS 2018 Workshop on Meta-Learning. [paper](https://arxiv.org/pdf/1806.06927) 

**Automatically learn inner vector learning rate, tune outer learning rate**

Alpha MAML: Adaptive Model-Agnostic Meta-Learning. ICML 2019 Workshop on Automated Machine Learning. [paper](https://arxiv.org/pdf/1905.07435) 

Meta-SGD: Learning to Learn Quickly for Few-Shot Learning. arXiv 2017. [paper](https://arxiv.org/pdf/1707.09835.pdf)

**Optimize only a subset of the parameters in the inner loop**

(DEML) Deep Meta-Learning: Learning to Learn in the Concept Space. arXiv 2018. [paper](https://arxiv.org/pdf/1802.03596)

(CAVIA) Fast Context Adaptation via Meta-Learning. ICML 2019. [paper](https://arxiv.org/pdf/1810.03642)

**Decouple inner learning rate, BN statistics per-step**

(MAML++) How to train your MAML. ICLR 2019. [paper](https://arxiv.org/pdf/1810.09502)

**Introduce context variables for increased expressive power**

(CAVIA) Fast Context Adaptation via Meta-Learning. ICML 2019. [paper](https://arxiv.org/pdf/1810.03642)

(Bias transformation) Meta-Learning and Universality: Deep Representations and Gradient Descent can Approximate any Learning Algorithm. ICLR 2018. [paper](https://arxiv.org/abs/1710.11622)

### Non-Parametric Methods via Metric Learning

Siamese Neural Networks for One-shot Image Recognition. ICML 2015. [paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

Matching Networks for One Shot Learning. NIPS 2016. [paper](https://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf)

Prototypical Networks for Few-shot Learning. NIPS 2017. [paper](https://arxiv.org/pdf/1703.05175)

**Learn non-linear relation module on embeddings**

Learning to Compare: Relation Network for Few-Shot Learning. CVPR 2018. [paper](https://arxiv.org/pdf/1711.06025)

**Learn infinite mixture of prototypes**

Infinite Mixture Prototypes for Few-Shot Learning. ICML 2019. [paper](https://arxiv.org/pdf/1902.04552.pdf)

**Perform message passing on embeddings**

Few-Shot Learning with Graph Neural Networks ICLR 2018. [paper](https://arxiv.org/pdf/1711.04043)

### Bayesian Meta-Learning & Generative Models

#### Amortized Inference

Amortized Bayesian Meta-Learning. ICLR 2019. [paper](https://openreview.net/pdf?id=rkgpy3C5tX)

#### Ensemble Method

Bayesian Model-Agnostic Meta-Learning. NIPS 2018. [paper](https://arxiv.org/pdf/1806.03836)

#### Sampling & Hybrid Inference

Probabilistic Model-Agnostic Meta-Learning. NIPS 2018. [paper](https://arxiv.org/pdf/1806.02817)

Meta-Learning Probabilistic Inference for Prediction. ICLR 2019. [paper](https://openreview.net/pdf?id=HkxStoC5F7)

### Hybrid meta-learning approaches

Meta-Learning with Latent Embedding Optimization. ICLR 2019. [paper](https://arxiv.org/pdf/1807.05960)

Fast Context Adaptation via Meta-Learning. ICML 2019. [paper](https://arxiv.org/pdf/1810.03642)

Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples. ICLR 2020. [paper](https://arxiv.org/pdf/1903.03096)

Few-Shot Learning with Graph Neural Networks. ICLR 2018. [paper](https://arxiv.org/pdf/1711.04043)

(CAML) Learning to Learn with Conditional Class Dependencies. ICLR 2019. [paper](https://openreview.net/pdf?id=BJfOXnActQ)

### Application

Meta-Learning for Low-Resource Neural Machine Translation. EMNLP 2018. [paper](https://arxiv.org/pdf/1808.08437)

Few-shot Autoregressive Density Estimation: Towards Learning to Learn Distributions. ICLR 2018. [paper](https://arxiv.org/pdf/1710.10304)

One-Shot Imitation Learning. NIPS 2017. [paper](https://arxiv.org/pdf/1703.07326.pdf)

Massively Multitask Networks for Drug Discovery. ICML 2015. [paper](https://arxiv.org/pdf/1502.02072.pdf)

